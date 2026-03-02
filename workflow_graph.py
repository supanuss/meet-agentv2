from __future__ import annotations

import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from html_renderer import (
    apply_react_official_theme,
    fallback_render_html,
    html_compliance_issues,
    strip_markdown_fences,
)
from image_processor import (
    group_manifest_by_topic,
    image_to_base64_data_uri,
    merge_partial_image_outputs,
    resolve_image_path,
)
from llm_client import LLMClient
from pipeline_utils import (
    PipelineConfig,
    PipelineError,
    build_topic_text,
    chunked,
    cosine,
    ensure_dir,
    hms_to_sec,
    load_json,
    reduce_agent1_maps,
    sanitize_kg_for_output,
    save_json,
    sec_to_hms,
    timeline_snippet_by_range,
    fill_template,
)
from prompts import (
    AGENT1_SYS,
    AGENT1_USR,
    AGENT2_REDUCE_SYS,
    AGENT2_REDUCE_USR,
    AGENT2_SYS,
    AGENT2_USR,
    AGENT25_REDUCE_SYS,
    AGENT25_REDUCE_USR,
    AGENT25_SYS,
    AGENT25_USR,
    AGENT3A_SYS,
    AGENT3A_USR,
    AGENT3B_SYS,
    AGENT3B_USR,
    AGENT4_EXEC_SYS,
    AGENT4_EXEC_USR,
    AGENT4_TOPIC_SYS,
    AGENT4_TOPIC_USR,
    AGENT5_SYS,
    AGENT5_USR,
    HTML_CSS_JS_BUNDLE,
)


class WorkflowState(TypedDict, total=False):
    run_id: str
    artifact_dir: str
    run_meta: dict[str, Any]
    transcript: dict[str, Any]
    config_data: dict[str, Any]
    ocr_data: dict[str, Any]
    segments: list[dict[str, Any]]
    captures: list[dict[str, Any]]
    resume_cleaned: dict[str, Any]
    resume_kg: dict[str, Any]
    cleaned: dict[str, Any]
    kg: dict[str, Any]
    topics: list[dict[str, Any]]
    topic_map: dict[str, Any]
    image_by_topic: dict[str, list[dict[str, Any]]]
    image_manifest_output: dict[str, Any]
    summaries: dict[str, Any]
    html: str


class MeetingWorkflow:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.llm = LLMClient(cfg)
        self.graph = self._build_graph().compile()

    def _artifact_path(self, state: WorkflowState, filename: str) -> Path:
        return Path(state["artifact_dir"]) / filename

    def _save_json_if_enabled(self, state: WorkflowState, filename: str, data: Any) -> None:
        if self.cfg.save_intermediate:
            save_json(self._artifact_path(state, filename), data)

    def _save_html_if_enabled(self, state: WorkflowState, filename: str, html: str) -> None:
        if self.cfg.save_intermediate:
            self._artifact_path(state, filename).write_text(html, encoding="utf-8")

    def _append_log(
        self,
        run_meta: dict[str, Any],
        artifact_dir: str | None,
        message: str,
        **fields: Any,
    ) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        field_text = " ".join(f"{k}={v}" for k, v in fields.items())
        line = f"[{ts}] {message}" + (f" | {field_text}" if field_text else "")
        print(line)
        run_meta.setdefault("runtime_logs", []).append(line)

        if self.cfg.save_intermediate and artifact_dir:
            log_path = Path(artifact_dir) / "runtime.log"
            with log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def _effective_workers(self, item_count: int) -> int:
        if item_count <= 1:
            return 1
        return max(1, min(self.cfg.pipeline_max_concurrency, item_count))

    def _agenda_sort_key(self, agenda_number: Any) -> tuple:
        raw = str(agenda_number or "").strip()
        if not raw:
            return (999999,)
        parts = []
        for part in raw.split("."):
            token = part.strip()
            if token.isdigit():
                parts.append(int(token))
                continue
            m = re.match(r"(\d+)", token)
            if m:
                parts.append(int(m.group(1)))
            else:
                parts.append(999999)
        return tuple(parts)

    def _is_container_agenda_item(
        self,
        item: dict[str, Any],
        all_agenda_numbers: set[str],
    ) -> bool:
        num = str(item.get("agenda_number", "") or "").strip()
        if not num:
            return False
        prefix = f"{num}."
        has_child = any(n != num and n.startswith(prefix) for n in all_agenda_numbers)
        if not has_child:
            return False

        title = re.sub(r"\s+", "", str(item.get("title", "") or ""))
        dept = re.sub(r"\s+", "", str(item.get("department", "") or ""))

        # Typical parent/container agendas are generic department headers.
        if not title:
            return True
        if dept and title == dept:
            return True
        if title.startswith("ฝ่าย"):
            return True
        if len(title) <= 14 and ("ฝ่าย" in title or "โกดัง" in title):
            return True
        return False

    def _sample_timeline_for_agent3b(
        self,
        timeline: list[dict[str, Any]],
        max_items: int = 300,
    ) -> list[dict[str, Any]]:
        rows = [x for x in timeline if isinstance(x, dict)]
        if len(rows) <= max_items:
            return rows
        n = len(rows)
        idxs: list[int] = []
        for i in range(max_items):
            idx = int(round(i * (n - 1) / max(1, max_items - 1)))
            if not idxs or idx != idxs[-1]:
                idxs.append(idx)
        return [rows[i] for i in idxs]

    def _topic_coverage_ratio(
        self,
        extracted_topics: list[dict[str, Any]],
        timeline: list[dict[str, Any]],
    ) -> float:
        if not extracted_topics or not timeline:
            return 0.0
        tl_secs = [
            hms_to_sec(str(x.get("timestamp_hms", "00:00:00")))
            for x in timeline
            if isinstance(x, dict)
        ]
        if not tl_secs:
            return 0.0
        meeting_start = min(tl_secs)
        meeting_end = max(tl_secs)
        meeting_span = max(1, meeting_end - meeting_start)

        starts: list[int] = []
        ends: list[int] = []
        for t in extracted_topics:
            if not isinstance(t, dict):
                continue
            st = hms_to_sec(str(t.get("start_timestamp", "00:00:00")))
            ed = hms_to_sec(str(t.get("end_timestamp", "00:00:00")))
            if ed < st:
                ed = st
            starts.append(st)
            ends.append(ed)
        if not starts:
            return 0.0
        covered_span = max(1, max(ends) - min(starts))
        return min(1.0, max(0.0, covered_span / meeting_span))

    def _agent3b_fallback_from_kg(self, topics: list[dict[str, Any]]) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for t in topics:
            if not isinstance(t, dict):
                continue
            topic_id = str(t.get("id", "") or "")
            name = str(t.get("name", "") or "").strip()
            if not topic_id or not name:
                continue
            st = hms_to_sec(str(t.get("start_timestamp", "00:00:00")))
            ed = hms_to_sec(str(t.get("end_timestamp", "00:00:00")))
            if ed < st:
                ed = st
            speakers = t.get("key_speakers", []) if isinstance(t.get("key_speakers"), list) else []
            slides = t.get("slide_timestamps", []) if isinstance(t.get("slide_timestamps"), list) else []
            decisions = t.get("decisions", []) if isinstance(t.get("decisions"), list) else []
            actions = t.get("action_items", []) if isinstance(t.get("action_items"), list) else []
            rows.append(
                {
                    "id": topic_id,
                    "name": name,
                    "department": str(t.get("department", "") or ""),
                    "start_sec": st,
                    "end_sec": ed,
                    "key_speakers": [str(x) for x in speakers if str(x).strip()],
                    "slide_timestamps": [str(x) for x in slides if str(x).strip()],
                    "decisions_count": len(decisions),
                    "actions_count": len(actions),
                }
            )
        if not rows:
            return {"extracted_topics": [], "topic_flow": "ไม่พบหัวข้อจากความรู้ที่สกัดได้"}

        rows.sort(key=lambda x: (x["start_sec"], x["end_sec"]))
        clusters: list[dict[str, Any]] = []
        cur: dict[str, Any] | None = None
        max_gap_sec = 180
        max_cluster_items = 3
        max_cluster_span_sec = 13 * 60

        for r in rows:
            if cur is None:
                cur = {
                    "start_sec": r["start_sec"],
                    "end_sec": r["end_sec"],
                    "items": [r],
                }
                continue
            gap = r["start_sec"] - int(cur["end_sec"])
            new_span = max(int(cur["end_sec"]), r["end_sec"]) - int(cur["start_sec"])
            if gap <= max_gap_sec and len(cur["items"]) < max_cluster_items and new_span <= max_cluster_span_sec:
                cur["items"].append(r)
                cur["end_sec"] = max(int(cur["end_sec"]), r["end_sec"])
            else:
                clusters.append(cur)
                cur = {
                    "start_sec": r["start_sec"],
                    "end_sec": r["end_sec"],
                    "items": [r],
                }
        if cur is not None:
            clusters.append(cur)

        extracted: list[dict[str, Any]] = []
        for idx, c in enumerate(clusters, start=1):
            items = c["items"]
            names = [str(x.get("name", "") or "").strip() for x in items if str(x.get("name", "") or "").strip()]
            title = names[0] if names else f"หัวข้อการประชุมช่วงที่ {idx}"
            subtitle = " / ".join(names[1:3]) if len(names) > 1 else ""
            depts = [str(x.get("department", "") or "").strip() for x in items if str(x.get("department", "") or "").strip()]
            dept = Counter(depts).most_common(1)[0][0] if depts else ""
            spks: list[str] = []
            slides: list[str] = []
            decisions_count = 0
            actions_count = 0
            for x in items:
                spks.extend(x.get("key_speakers", []))
                slides.extend(x.get("slide_timestamps", []))
                decisions_count += int(x.get("decisions_count", 0) or 0)
                actions_count += int(x.get("actions_count", 0) or 0)
            uniq_spks = list(dict.fromkeys([s for s in spks if s]))[:8]
            uniq_slides = list(dict.fromkeys([s for s in slides if s]))[:8]
            dur_min = round(max(0, int(c["end_sec"]) - int(c["start_sec"])) / 60.0, 2)
            if "รายงาน" in title:
                topic_type = "report"
            elif "แจ้ง" in title or "ประกาศ" in title:
                topic_type = "announcement"
            elif decisions_count > 0:
                topic_type = "decision"
            else:
                topic_type = "discussion"
            importance = "high" if (dur_min >= 10 or decisions_count > 0 or actions_count > 0) else "medium"
            extracted.append(
                {
                    "id": str(items[0].get("id", f"T{idx:03}")),
                    "number": str(idx),
                    "title": title,
                    "subtitle": subtitle,
                    "department": dept,
                    "start_timestamp": sec_to_hms(int(c["start_sec"])),
                    "end_timestamp": sec_to_hms(int(c["end_sec"])),
                    "duration_minutes": dur_min,
                    "topic_type": topic_type,
                    "key_speakers": uniq_spks,
                    "slide_timestamps": uniq_slides,
                    "importance": importance,
                }
            )

        topic_flow = (
            f"สรุปอัตโนมัติจาก KG โดยจัดกลุ่มตามช่วงเวลา ได้ {len(extracted)} หัวข้อ "
            f"ครอบคลุมตั้งแต่ {extracted[0]['start_timestamp']} ถึง {extracted[-1]['end_timestamp']}"
        )
        return {"extracted_topics": extracted, "topic_flow": topic_flow}

    def _agent3a_fallback_from_hints(
        self,
        agenda_lines: list[str],
        semantic_hints: list[dict[str, Any]],
        topics: list[dict[str, Any]],
    ) -> dict[str, Any]:
        topic_by_id: dict[str, dict[str, Any]] = {}
        for topic in topics:
            if not isinstance(topic, dict):
                continue
            tid = str(topic.get("id", "") or "").strip()
            if tid:
                topic_by_id[tid] = topic

        agenda_mapping: list[dict[str, Any]] = []
        discussed = 0
        for idx, line in enumerate(agenda_lines, start=1):
            hint = semantic_hints[idx - 1] if idx - 1 < len(semantic_hints) else {}
            best_topic_id = str(hint.get("semantic_best_topic", "") or "").strip()
            topic = topic_by_id.get(best_topic_id)

            agenda_number = str(idx)
            agenda_title = str(line or "").strip()
            m = re.match(r"^\s*(\d+(?:\.\d+)*)[\).:\-]?\s*(.*)$", agenda_title)
            if m:
                agenda_number = m.group(1).strip() or agenda_number
                parsed_title = m.group(2).strip()
                if parsed_title:
                    agenda_title = parsed_title

            department = ""
            dep_match = re.search(r"(ฝ่าย[^\-\|:]{2,40})", agenda_title)
            if dep_match:
                department = dep_match.group(1).strip()

            mapped_topics: list[str] = []
            tr = {"start": "00:00:00", "end": "00:00:00"}
            key_speaker = ""
            if topic:
                mapped_topics = [best_topic_id]
                tr = {
                    "start": str(topic.get("start_timestamp", "00:00:00") or "00:00:00"),
                    "end": str(topic.get("end_timestamp", "00:00:00") or "00:00:00"),
                }
                speakers = topic.get("key_speakers", [])
                if isinstance(speakers, list) and speakers:
                    key_speaker = str(speakers[0] or "")
                discussed += 1

            agenda_mapping.append(
                {
                    "agenda_number": agenda_number,
                    "agenda_title": agenda_title,
                    "agenda_department": department,
                    "status": "discussed" if mapped_topics else "not_discussed",
                    "mapped_topics": mapped_topics,
                    "time_range": tr,
                    "key_speaker": key_speaker,
                }
            )

        total = len(agenda_mapping)
        return {
            "agenda_mapping": agenda_mapping,
            "coverage_stats": {
                "total": total,
                "discussed": discussed,
                "not_discussed": max(0, total - discussed),
            },
        }

    def _coerce_time_to_hms(self, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            sec = max(int(float(value)), 0)
            return sec_to_hms(sec)
        text = str(value).strip()
        if not text:
            return None
        if ":" in text:
            return sec_to_hms(hms_to_sec(text))
        try:
            sec = max(int(float(text)), 0)
            return sec_to_hms(sec)
        except Exception:
            return None

    def _extract_topic_ref_number(self, topic_ref: str) -> str:
        text = str(topic_ref or "").strip()
        if not text:
            return ""
        m = re.search(r"(\d+(?:\.\d+)*)", text)
        return m.group(1) if m else ""

    def _apply_topic_time_overrides(
        self,
        topic_map: dict[str, Any],
        config_data: dict[str, Any],
    ) -> int:
        overrides = config_data.get("TOPIC_TIME_OVERRIDES", [])
        if not isinstance(overrides, list) or not overrides:
            return 0

        applied = 0

        def match_override(ref_text: str, ref_num: str, item_num: str, item_title: str) -> bool:
            num = str(item_num or "").strip()
            num_norm = self._extract_topic_ref_number(num) or num
            title = str(item_title or "").strip().lower()
            if ref_num and (num == ref_num or num_norm == ref_num):
                return True
            rt = ref_text.lower().strip()
            if not rt:
                return False
            if rt == num.lower():
                return True
            if ref_num and num_norm == ref_num:
                return True
            if rt in title:
                return True
            return False

        if "agenda_mapping" in topic_map and isinstance(topic_map.get("agenda_mapping"), list):
            for ov in overrides:
                if not isinstance(ov, dict):
                    continue
                ref = str(ov.get("topic", "") or ov.get("agenda", "") or ov.get("agenda_number", "")).strip()
                start_hms = self._coerce_time_to_hms(ov.get("start_time"))
                end_hms = self._coerce_time_to_hms(ov.get("end_time"))
                if not ref or not start_hms or not end_hms:
                    continue
                start_sec = hms_to_sec(start_hms)
                end_sec = hms_to_sec(end_hms)
                if end_sec < start_sec:
                    end_sec = start_sec
                    end_hms = start_hms
                ref_num = self._extract_topic_ref_number(ref)

                for item in topic_map.get("agenda_mapping", []):
                    if not isinstance(item, dict):
                        continue
                    if not match_override(
                        ref_text=ref,
                        ref_num=ref_num,
                        item_num=str(
                            item.get("agenda_number", "")
                            or item.get("topic_ref", "")
                            or item.get("number", "")
                            or ""
                        ),
                        item_title=str(item.get("agenda_title", "") or item.get("title", "") or ""),
                    ):
                        continue
                    item["time_range"] = {"start": start_hms, "end": end_hms}
                    item["_time_range_overridden"] = True
                    applied += 1
                    break

        if "extracted_topics" in topic_map and isinstance(topic_map.get("extracted_topics"), list):
            for ov in overrides:
                if not isinstance(ov, dict):
                    continue
                ref = str(ov.get("topic", "") or ov.get("agenda", "") or ov.get("agenda_number", "")).strip()
                start_hms = self._coerce_time_to_hms(ov.get("start_time"))
                end_hms = self._coerce_time_to_hms(ov.get("end_time"))
                if not ref or not start_hms or not end_hms:
                    continue
                start_sec = hms_to_sec(start_hms)
                end_sec = hms_to_sec(end_hms)
                if end_sec < start_sec:
                    end_sec = start_sec
                    end_hms = start_hms
                ref_num = self._extract_topic_ref_number(ref)

                for item in topic_map.get("extracted_topics", []):
                    if not isinstance(item, dict):
                        continue
                    if not match_override(
                        ref_text=ref,
                        ref_num=ref_num,
                        item_num=str(
                            item.get("number", "")
                            or item.get("topic_ref", "")
                            or item.get("agenda_number", "")
                            or ""
                        ),
                        item_title=str(
                            item.get("title", "")
                            or item.get("topic_title", "")
                            or item.get("agenda_title", "")
                            or ""
                        ),
                    ):
                        continue
                    item["start_timestamp"] = start_hms
                    item["end_timestamp"] = end_hms
                    item["duration_minutes"] = round(max(0, end_sec - start_sec) / 60.0, 2)
                    item["_time_range_overridden"] = True
                    applied += 1
                    break

        return applied

    def _remove_stutter(self, text: str) -> str:
        tokens = str(text or "").strip().split()
        if not tokens:
            return ""
        out = [tokens[0]]
        for tok in tokens[1:]:
            if tok != out[-1]:
                out.append(tok)
        return " ".join(out)

    def _build_slides_from_ocr(self, ocr_subset: list[dict[str, Any]]) -> list[dict[str, Any]]:
        slides: list[dict[str, Any]] = []
        for c in ocr_subset:
            ocr_text = str(c.get("ocr_text", "") or "")
            lines = [ln.strip() for ln in ocr_text.splitlines() if ln.strip()]
            title = lines[0] if lines else ""
            slides.append(
                {
                    "timestamp_hms": str(c.get("timestamp_hms", "00:00:00")),
                    "image_path": str(c.get("image_path", "")),
                    "ocr_text": ocr_text,
                    "has_table": "<table" in ocr_text.lower(),
                    "has_figure": "<figure" in ocr_text.lower(),
                    "title": title,
                }
            )
        return slides

    def _nearest_slide_context(self, ts: float, slides: list[dict[str, Any]]) -> str | None:
        best_text: str | None = None
        best_d = 10**9
        for s in slides:
            sec = hms_to_sec(str(s.get("timestamp_hms", "00:00:00")))
            d = abs(sec - int(ts))
            if d <= 60 and d < best_d:
                best_d = d
                best_text = str(s.get("ocr_text", "") or "")
        return best_text

    def _compact_ocr_for_agent1(self, ocr_subset: list[dict[str, Any]]) -> list[dict[str, Any]]:
        compact: list[dict[str, Any]] = []
        max_snippet_chars = max(120, int(self.cfg.agent1_ocr_snippet_chars))

        for idx, cap in enumerate(ocr_subset, start=1):
            raw = str(cap.get("ocr_text", "") or "")
            lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
            title = lines[0] if lines else ""
            has_table = "<table" in raw.lower()
            has_figure = "<figure" in raw.lower()

            text_no_html = re.sub(r"<[^>]+>", " ", raw)
            text_no_html = re.sub(r"\s+", " ", text_no_html).strip()
            snippet = text_no_html[:max_snippet_chars]
            if len(text_no_html) > max_snippet_chars:
                snippet += "..."
            # Agent1 only needs lightweight slide context; avoid pushing heavy table body text.
            if has_table and title:
                snippet = title[:max_snippet_chars]

            compact.append(
                {
                    "capture_index": int(cap.get("capture_index", idx) or idx),
                    "timestamp_hms": str(cap.get("timestamp_hms", "00:00:00") or "00:00:00"),
                    "timestamp_sec": float(cap.get("timestamp_sec", 0) or 0),
                    "image_path": str(cap.get("image_path", "") or ""),
                    "ocr_file_size_bytes": int(cap.get("ocr_file_size_bytes", 0) or 0),
                    "ocr_skipped_reason": str(cap.get("ocr_skipped_reason", "") or ""),
                    "has_table": has_table,
                    "has_figure": has_figure,
                    "title": title[:220],
                    # Keep ocr_text key for prompt compatibility, but send only compact snippet.
                    "ocr_text": snippet,
                }
            )

        return compact

    def _chunk_text_by_chars(self, text: str, chunk_size: int = 1000, overlap_ratio: float = 0.10) -> list[str]:
        t = str(text or "")
        if not t:
            return [""]
        size = max(100, int(chunk_size))
        overlap = int(size * max(0.0, min(overlap_ratio, 0.9)))
        step = max(1, size - overlap)
        if len(t) <= size:
            return [t]
        out: list[str] = []
        i = 0
        while i < len(t):
            part = t[i : i + size]
            if not part:
                break
            out.append(part)
            if i + size >= len(t):
                break
            i += step
        return out

    def _build_ocr_only_payload_for_agent1(self, ocr_subset: list[dict[str, Any]]) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for idx, cap in enumerate(ocr_subset, start=1):
            raw = str(cap.get("ocr_text", "") or "")
            lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
            title = lines[0] if lines else ""
            has_table = "<table" in raw.lower()
            has_figure = "<figure" in raw.lower()

            plain = re.sub(r"<[^>]+>", " ", raw)
            plain = re.sub(r"\s+", " ", plain).strip()
            text_chunks = self._chunk_text_by_chars(plain, chunk_size=1000, overlap_ratio=0.10)

            total = len(text_chunks)
            for cidx, chunk_text in enumerate(text_chunks, start=1):
                payload.append(
                    {
                        "capture_index": int(cap.get("capture_index", idx) or idx),
                        "chunk_index": cidx,
                        "chunk_total": total,
                        "timestamp_hms": str(cap.get("timestamp_hms", "00:00:00") or "00:00:00"),
                        "timestamp_sec": float(cap.get("timestamp_sec", 0) or 0),
                        "image_path": str(cap.get("image_path", "") or ""),
                        "ocr_file_size_bytes": int(cap.get("ocr_file_size_bytes", 0) or 0),
                        "ocr_skipped_reason": str(cap.get("ocr_skipped_reason", "") or ""),
                        "has_table": has_table,
                        "has_figure": has_figure,
                        "title": title[:220],
                        "ocr_text": chunk_text,
                    }
                )
        return payload

    def _select_agent1_ocr_subset(
        self,
        captures: list[dict[str, Any]],
        start_sec: float,
        end_sec: float,
        window_sec: float = 120.0,
    ) -> list[dict[str, Any]]:
        in_window = [
            c
            for c in captures
            if (start_sec - window_sec) <= float(c.get("timestamp_sec", 0) or 0) <= (end_sec + window_sec)
        ]
        valid = [c for c in in_window if not str(c.get("ocr_skipped_reason", "") or "").strip()]
        if not valid:
            valid = in_window

        mid = (start_sec + end_sec) / 2.0
        valid.sort(key=lambda c: abs(float(c.get("timestamp_sec", 0) or 0) - mid))
        max_caps = max(1, int(self.cfg.agent1_ocr_max_captures))
        return valid[:max_caps]

    def _agent1_chunk_fallback(
        self,
        seg_chunk: list[dict[str, Any]],
        ocr_subset: list[dict[str, Any]],
    ) -> dict[str, Any]:
        slides = self._build_slides_from_ocr(ocr_subset)
        timeline: list[dict[str, Any]] = []

        for seg in seg_chunk:
            start_sec = float(seg.get("start", 0) or 0)
            text = self._remove_stutter(str(seg.get("text", "") or ""))
            text = re.sub(r"\s+", " ", text).strip()
            if not text:
                continue
            timeline.append(
                {
                    "timestamp_sec": start_sec,
                    "timestamp_hms": sec_to_hms(start_sec),
                    "speaker": str(seg.get("speaker", "UNKNOWN") or "UNKNOWN"),
                    "text": text,
                    "slide_context": self._nearest_slide_context(start_sec, slides),
                }
            )

        return {
            "meeting_meta": {
                "title": "",
                "date": "",
                "time_range": "",
                "platform": "ZOOM",
                "company": "บริษัทแสงฟ้าก่อสร้าง จำกัด",
                "chairperson": "",
                "attendees": [],
            },
            "timeline": timeline,
            "slides": slides,
        }

    def _agent1_call_llm(
        self,
        seg_chunk: list[dict[str, Any]],
        ocr_subset: list[dict[str, Any]],
        config_data: dict[str, Any],
        tag: str,
    ) -> dict[str, Any]:
        compact_ocr = self._compact_ocr_for_agent1(ocr_subset)
        user = fill_template(
            AGENT1_USR,
            TRANSCRIPT=json.dumps({"segments": seg_chunk}, ensure_ascii=False),
            OCR=json.dumps({"captures": compact_ocr}, ensure_ascii=False),
            CONFIG=json.dumps(config_data, ensure_ascii=False),
        )
        out = self.llm.call(
            AGENT1_SYS,
            user,
            json_mode=True,
            # Validate/repair timeline in-code to avoid hard-fail on missing key.
            required_keys=[],
            tag=tag,
        )
        assert isinstance(out, dict)
        if not isinstance(out.get("meeting_meta"), dict):
            out["meeting_meta"] = {}
        if not isinstance(out.get("slides"), list):
            out["slides"] = []
        has_text = any(str(seg.get("text", "") or "").strip() for seg in seg_chunk)
        if not isinstance(out.get("timeline"), list) or (has_text and not out.get("timeline")):
            repaired = self._agent1_chunk_fallback(seg_chunk, ocr_subset)
            # Keep useful meta/slides from LLM output when present.
            llm_meta = out.get("meeting_meta", {})
            if isinstance(llm_meta, dict) and llm_meta:
                base_meta = repaired.get("meeting_meta", {})
                if not isinstance(base_meta, dict):
                    base_meta = {}
                merged_meta = dict(base_meta)
                merged_meta.update({k: v for k, v in llm_meta.items() if v not in ("", [], None)})
                repaired["meeting_meta"] = merged_meta
            llm_slides = out.get("slides", [])
            if isinstance(llm_slides, list) and llm_slides:
                repaired["slides"] = llm_slides
            return repaired
        return out

    def _agent1_call_llm_ocr_only(
        self,
        ocr_subset: list[dict[str, Any]],
        config_data: dict[str, Any],
        tag: str,
    ) -> dict[str, Any]:
        compact_ocr = self._build_ocr_only_payload_for_agent1(ocr_subset)
        user = fill_template(
            AGENT1_USR,
            TRANSCRIPT=json.dumps({"segments": []}, ensure_ascii=False),
            OCR=json.dumps({"captures": compact_ocr}, ensure_ascii=False),
            CONFIG=json.dumps(config_data, ensure_ascii=False),
        )
        out = self.llm.call(
            AGENT1_SYS,
            user,
            json_mode=True,
            # OCR-only path uses slides/meta opportunistically; avoid hard-fail on partial JSON.
            required_keys=[],
            tag=tag,
        )
        assert isinstance(out, dict)
        if not isinstance(out.get("meeting_meta"), dict):
            out["meeting_meta"] = {}
        if not isinstance(out.get("timeline"), list):
            out["timeline"] = []
        if not isinstance(out.get("slides"), list):
            out["slides"] = []
        return out

    def _agent1_call_llm_transcript_only(
        self,
        seg_chunk: list[dict[str, Any]],
        config_data: dict[str, Any],
        tag: str,
    ) -> dict[str, Any]:
        user = fill_template(
            AGENT1_USR,
            TRANSCRIPT=json.dumps({"segments": seg_chunk}, ensure_ascii=False),
            OCR=json.dumps({"captures": []}, ensure_ascii=False),
            CONFIG=json.dumps(config_data, ensure_ascii=False),
        )
        out = self.llm.call(
            AGENT1_SYS,
            user,
            json_mode=True,
            # Transcript-only path primarily needs timeline; meta/slides can be filled with defaults.
            required_keys=[],
            tag=tag,
        )
        assert isinstance(out, dict)
        if not isinstance(out.get("meeting_meta"), dict):
            out["meeting_meta"] = {}
        if not isinstance(out.get("slides"), list):
            out["slides"] = []
        has_text = any(str(seg.get("text", "") or "").strip() for seg in seg_chunk)
        if not isinstance(out.get("timeline"), list) or (has_text and not out.get("timeline")):
            repaired = self._agent1_chunk_fallback(seg_chunk, [])
            llm_meta = out.get("meeting_meta", {})
            if isinstance(llm_meta, dict) and llm_meta:
                base_meta = repaired.get("meeting_meta", {})
                if not isinstance(base_meta, dict):
                    base_meta = {}
                merged_meta = dict(base_meta)
                merged_meta.update({k: v for k, v in llm_meta.items() if v not in ("", [], None)})
                repaired["meeting_meta"] = merged_meta
            llm_slides = out.get("slides", [])
            if isinstance(llm_slides, list) and llm_slides:
                repaired["slides"] = llm_slides
            return repaired
        return out

    def _agent1_subchunk_recover(
        self,
        seg_chunk: list[dict[str, Any]],
        ocr_subset: list[dict[str, Any]],
        config_data: dict[str, Any],
        run_meta: dict[str, Any],
        artifact_dir: str | None,
        parent_chunk_idx: int,
        parent_chunk_total: int,
    ) -> dict[str, Any]:
        sub_size = max(20, min(self.cfg.agent1_subchunk_size, len(seg_chunk)))
        sub_chunks = chunked(seg_chunk, sub_size, overlap=0)
        self._append_log(
            run_meta,
            artifact_dir,
            "Agent1 subchunk start",
            parent=f"{parent_chunk_idx}/{parent_chunk_total}",
            subchunks=len(sub_chunks),
            sub_size=sub_size,
        )

        partials: list[dict[str, Any]] = []
        for sidx, sub in enumerate(sub_chunks, start=1):
            s_start = float(sub[0].get("start", 0) or 0)
            s_end = float(sub[-1].get("end", s_start) or s_start)
            sub_ocr = self._select_agent1_ocr_subset(ocr_subset, s_start, s_end, window_sec=120.0)
            self._append_log(
                run_meta,
                artifact_dir,
                "Agent1 subchunk",
                parent=f"{parent_chunk_idx}/{parent_chunk_total}",
                chunk=f"{sidx}/{len(sub_chunks)}",
                seg=len(sub),
                ocr=len(sub_ocr),
                start=sec_to_hms(s_start),
                end=sec_to_hms(s_end),
            )
            try:
                out = self._agent1_call_llm(
                    sub,
                    sub_ocr,
                    config_data,
                    tag=f"agent1_subchunk_{parent_chunk_idx}_{sidx}",
                )
            except Exception as sub_exc:
                self._append_log(
                    run_meta,
                    artifact_dir,
                    "Agent1 subchunk fallback",
                    parent=f"{parent_chunk_idx}/{parent_chunk_total}",
                    chunk=f"{sidx}/{len(sub_chunks)}",
                    error=str(sub_exc),
                )
                out = self._agent1_chunk_fallback(sub, sub_ocr)
            partials.append(out)

        merged = reduce_agent1_maps(partials, config_data)
        self._append_log(
            run_meta,
            artifact_dir,
            "Agent1 subchunk done",
            parent=f"{parent_chunk_idx}/{parent_chunk_total}",
            timeline=len(merged.get("timeline", [])),
            slides=len(merged.get("slides", [])),
        )
        return merged

    def _agent1_transcript_split_recover(
        self,
        seg_chunk: list[dict[str, Any]],
        config_data: dict[str, Any],
        run_meta: dict[str, Any],
        artifact_dir: str | None,
        parent_chunk_idx: int,
        parent_chunk_total: int,
    ) -> dict[str, Any]:
        if len(seg_chunk) <= 1:
            return self._agent1_chunk_fallback(seg_chunk, [])

        split_at = max(1, len(seg_chunk) // 2)
        parts = [seg_chunk[:split_at], seg_chunk[split_at:]]
        parts = [p for p in parts if p]

        self._append_log(
            run_meta,
            artifact_dir,
            "Agent1 transcript split start",
            parent=f"{parent_chunk_idx}/{parent_chunk_total}",
            subchunks=len(parts),
            split_at=split_at,
        )

        partials: list[dict[str, Any]] = []
        for sidx, sub in enumerate(parts, start=1):
            s_start = float(sub[0].get("start", 0) or 0)
            s_end = float(sub[-1].get("end", s_start) or s_start)
            self._append_log(
                run_meta,
                artifact_dir,
                "Agent1 transcript split chunk",
                parent=f"{parent_chunk_idx}/{parent_chunk_total}",
                chunk=f"{sidx}/{len(parts)}",
                seg=len(sub),
                start=sec_to_hms(s_start),
                end=sec_to_hms(s_end),
            )
            try:
                out = self._agent1_call_llm_transcript_only(
                    seg_chunk=sub,
                    config_data=config_data,
                    tag=f"agent1_transcript_subchunk_{parent_chunk_idx}_{sidx}",
                )
            except Exception as sub_exc:
                self._append_log(
                    run_meta,
                    artifact_dir,
                    "Agent1 transcript split fallback",
                    parent=f"{parent_chunk_idx}/{parent_chunk_total}",
                    chunk=f"{sidx}/{len(parts)}",
                    error=str(sub_exc),
                )
                out = self._agent1_chunk_fallback(sub, [])
            partials.append(out)

        merged = reduce_agent1_maps(partials, config_data)
        self._append_log(
            run_meta,
            artifact_dir,
            "Agent1 transcript split done",
            parent=f"{parent_chunk_idx}/{parent_chunk_total}",
            timeline=len(merged.get("timeline", [])),
        )
        return merged

    def _normalize_topic(self, raw: Any, idx: int) -> dict[str, Any] | None:
        if not isinstance(raw, dict):
            return None
        start_hms = str(raw.get("start_timestamp", "") or "00:00:00")
        end_hms = str(raw.get("end_timestamp", "") or start_hms)
        start_sec = hms_to_sec(start_hms)
        end_sec = hms_to_sec(end_hms)
        if end_sec < start_sec:
            end_sec = start_sec
            end_hms = start_hms

        duration = raw.get("duration_minutes", 0)
        try:
            duration_minutes = int(duration)
        except Exception:
            duration_minutes = max((end_sec - start_sec) // 60, 0)
        duration_minutes = max(duration_minutes, 0)

        name = str(raw.get("name", "") or raw.get("title", "") or f"หัวข้อที่ {idx}")
        topic_id = str(raw.get("id", "") or f"T{idx:03d}")
        if not topic_id.startswith("T"):
            topic_id = f"T{idx:03d}"

        def list_of_str(val: Any) -> list[str]:
            if not isinstance(val, list):
                return []
            return [str(x) for x in val if str(x).strip()]

        return {
            "id": topic_id,
            "name": name,
            "department": str(raw.get("department", "") or ""),
            "start_timestamp": start_hms,
            "end_timestamp": end_hms,
            "duration_minutes": duration_minutes,
            "key_speakers": list_of_str(raw.get("key_speakers")),
            "slide_timestamps": list_of_str(raw.get("slide_timestamps")),
            "summary_points": list_of_str(raw.get("summary_points"))[:5],
            "issues": list_of_str(raw.get("issues")),
            "decisions": list_of_str(raw.get("decisions")),
            "action_items": list_of_str(raw.get("action_items")),
        }

    def _merge_agent2_entities(self, partial_kgs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        keys = ["people", "projects", "equipment", "financials", "issues", "decisions", "action_items"]
        out: dict[str, list[dict[str, Any]]] = {k: [] for k in keys}
        seen: dict[str, set[str]] = {k: set() for k in keys}

        for partial in partial_kgs:
            entities = partial.get("entities", {})
            if not isinstance(entities, dict):
                continue
            for key in keys:
                values = entities.get(key, [])
                if not isinstance(values, list):
                    continue
                for item in values:
                    if not isinstance(item, dict):
                        continue
                    sig = json.dumps(item, ensure_ascii=False, sort_keys=True)
                    if sig in seen[key]:
                        continue
                    seen[key].add(sig)
                    out[key].append(item)
        return out

    def _empty_agent2_entities(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "people": [],
            "projects": [],
            "equipment": [],
            "financials": [],
            "issues": [],
            "decisions": [],
            "action_items": [],
        }

    def _agent2_chunk_fallback(
        self,
        tl_chunk: list[dict[str, Any]],
        slides_subset: list[dict[str, Any]],
        chunk_idx: int,
    ) -> dict[str, Any]:
        if not tl_chunk:
            return {"entities": self._empty_agent2_entities(), "topics": []}

        start_sec = int(float(tl_chunk[0].get("timestamp_sec", 0) or 0))
        end_sec = int(float(tl_chunk[-1].get("timestamp_sec", start_sec) or start_sec))
        if end_sec < start_sec:
            end_sec = start_sec

        speakers = Counter(str(r.get("speaker", "UNKNOWN") or "UNKNOWN") for r in tl_chunk if isinstance(r, dict))
        key_speakers = [name for name, _ in speakers.most_common(3)]

        summary_points: list[str] = []
        for row in tl_chunk:
            if not isinstance(row, dict):
                continue
            text = str(row.get("text", "") or "").strip()
            if not text:
                continue
            trimmed = text[:180]
            if trimmed in summary_points:
                continue
            summary_points.append(trimmed)
            if len(summary_points) >= 4:
                break

        slide_timestamps: list[str] = []
        for slide in slides_subset:
            if not isinstance(slide, dict):
                continue
            ts = str(slide.get("timestamp_hms", "") or "")
            if not ts or ts in slide_timestamps:
                continue
            slide_timestamps.append(ts)
            if len(slide_timestamps) >= 8:
                break

        topic = {
            "id": f"F{chunk_idx:03d}",
            "name": f"หัวข้อช่วง {sec_to_hms(start_sec)} - {sec_to_hms(end_sec)}",
            "department": "",
            "start_timestamp": sec_to_hms(start_sec),
            "end_timestamp": sec_to_hms(end_sec),
            "duration_minutes": max((end_sec - start_sec) // 60, 1),
            "key_speakers": key_speakers,
            "slide_timestamps": slide_timestamps,
            "summary_points": summary_points,
            "issues": [],
            "decisions": [],
            "action_items": [],
        }
        return {"entities": self._empty_agent2_entities(), "topics": [topic]}

    def _collect_topics_from_partials(self, partial_kgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        topics: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()

        for partial in partial_kgs:
            raw_topics = partial.get("topics", [])
            if not isinstance(raw_topics, list):
                continue
            for raw in raw_topics:
                normalized = self._normalize_topic(raw, len(topics) + 1)
                if not normalized:
                    continue
                sig = (
                    normalized["name"],
                    normalized["start_timestamp"],
                    normalized["end_timestamp"],
                )
                if sig in seen:
                    continue
                seen.add(sig)
                topics.append(normalized)

        for i, topic in enumerate(topics, start=1):
            topic["id"] = f"T{i:03d}"
        return topics

    def _synthesize_topics_from_timeline(self, cleaned: dict[str, Any]) -> list[dict[str, Any]]:
        timeline = cleaned.get("timeline", [])
        if not isinstance(timeline, list) or not timeline:
            return []
        slides = cleaned.get("slides", []) if isinstance(cleaned.get("slides"), list) else []

        min_sec = int(float(timeline[0].get("timestamp_sec", 0) or 0))
        max_sec = int(float(timeline[-1].get("timestamp_sec", min_sec) or min_sec))
        if max_sec < min_sec:
            max_sec = min_sec

        window_sec = 12 * 60
        out: list[dict[str, Any]] = []
        cursor = min_sec

        while cursor <= max_sec and len(out) < 20:
            end_sec = min(cursor + window_sec - 1, max_sec)
            rows = [
                r for r in timeline if cursor <= int(float(r.get("timestamp_sec", 0) or 0)) <= end_sec
            ]
            if not rows:
                cursor = end_sec + 1
                continue

            speakers = Counter(str(r.get("speaker", "UNKNOWN") or "UNKNOWN") for r in rows)
            key_speakers = [name for name, _ in speakers.most_common(3)]

            summary_points: list[str] = []
            for row in rows:
                text = str(row.get("text", "") or "").strip()
                if not text:
                    continue
                if text in summary_points:
                    continue
                summary_points.append(text[:180])
                if len(summary_points) >= 4:
                    break

            slide_ts: list[str] = []
            for s in slides:
                ts = str(s.get("timestamp_hms", "00:00:00"))
                sec = hms_to_sec(ts)
                if cursor <= sec <= end_sec:
                    slide_ts.append(ts)

            idx = len(out) + 1
            out.append(
                {
                    "id": f"T{idx:03d}",
                    "name": f"สรุปการประชุมช่วง {sec_to_hms(cursor)} - {sec_to_hms(end_sec)}",
                    "department": "",
                    "start_timestamp": sec_to_hms(cursor),
                    "end_timestamp": sec_to_hms(end_sec),
                    "duration_minutes": max((end_sec - cursor + 1) // 60, 1),
                    "key_speakers": key_speakers,
                    "slide_timestamps": slide_ts[:8],
                    "summary_points": summary_points,
                    "issues": [],
                    "decisions": [],
                    "action_items": [],
                }
            )
            cursor = end_sec + 1

        return out

    def _agent2_deterministic_fallback(
        self,
        partial_kgs: list[dict[str, Any]],
        cleaned: dict[str, Any],
    ) -> dict[str, Any]:
        entities = self._merge_agent2_entities(partial_kgs)
        topics = self._collect_topics_from_partials(partial_kgs)
        if not topics:
            topics = self._synthesize_topics_from_timeline(cleaned)
        return {"entities": entities, "topics": topics}

    def _agent25_call_llm(
        self,
        cap_chunk: list[dict[str, Any]],
        topic_no_vec: list[dict[str, Any]],
        *,
        tag: str,
    ) -> dict[str, Any]:
        user = fill_template(
            AGENT25_USR,
            CAPTURES=json.dumps(cap_chunk, ensure_ascii=False),
            KG_TOPICS=json.dumps(topic_no_vec, ensure_ascii=False),
        )
        out = self.llm.call(
            AGENT25_SYS,
            user,
            json_mode=True,
            required_keys=["image_manifest", "statistics"],
            tag=tag,
        )
        assert isinstance(out, dict)
        return out

    def _agent25_match_topic(
        self,
        timestamp_sec: float,
        ocr_text: str,
        topics: list[dict[str, Any]],
    ) -> tuple[str, str]:
        t_sec = int(max(timestamp_sec, 0))
        for topic in topics:
            if not isinstance(topic, dict):
                continue
            tid = str(topic.get("id", "") or "")
            tname = str(topic.get("name", "") or "")
            start_hms = str(topic.get("start_timestamp", "00:00:00") or "00:00:00")
            end_hms = str(topic.get("end_timestamp", start_hms) or start_hms)
            start_sec = hms_to_sec(start_hms)
            end_sec = hms_to_sec(end_hms)
            if start_sec <= t_sec <= max(end_sec, start_sec):
                return tid, tname

        lowered = ocr_text.lower()
        best: tuple[int, str, str] = (-1, "", "")
        for topic in topics:
            if not isinstance(topic, dict):
                continue
            tid = str(topic.get("id", "") or "")
            tname = str(topic.get("name", "") or "")
            tokens = [tok for tok in re.findall(r"[A-Za-z0-9ก-๙]{3,}", tname.lower()) if len(tok) >= 3]
            score = sum(1 for tok in set(tokens) if tok and tok in lowered)
            if score > best[0]:
                best = (score, tid, tname)

        if best[0] > 0:
            return best[1], best[2]
        return "", ""

    def _agent25_chunk_fallback(
        self,
        cap_chunk: list[dict[str, Any]],
        topic_no_vec: list[dict[str, Any]],
    ) -> dict[str, Any]:
        manifest: list[dict[str, Any]] = []
        by_type: Counter[str] = Counter()
        filtered = 0

        for cap in cap_chunk:
            ocr_text = str(cap.get("ocr_text", "") or "")
            lowered = ocr_text.lower()
            capture_index = int(cap.get("capture_index", 0) or 0)
            timestamp_sec = float(cap.get("timestamp_sec", 0) or 0)
            timestamp_hms = str(cap.get("timestamp_hms", sec_to_hms(timestamp_sec)) or sec_to_hms(timestamp_sec))
            image_path = str(cap.get("image_path", "") or "")
            file_size = int(cap.get("ocr_file_size_bytes", 0) or 0)
            skipped_reason = str(cap.get("ocr_skipped_reason", "") or "").strip()

            is_filtered = bool(skipped_reason) or file_size < 30000 or not ocr_text.strip()
            if is_filtered:
                filtered += 1
                by_type["SKIPPED"] += 1
                continue

            if "<table" in lowered:
                content_type = "DATA_TABLE"
            elif "<figure" in lowered and any(k in lowered for k in ["chart", "graph", "แผนภูมิ", "กราฟ"]):
                content_type = "CHART"
            elif "<figure" in lowered:
                content_type = "PHOTO"
            elif any(k in lowered for k in [".pdf", ".ppt", ".pptx", "adobe acrobat", "powerpoint"]):
                content_type = "DOCUMENT"
            elif "zoom" in lowered and any(k in lowered for k in ["participant", "gallery", "ผู้เข้าร่วม"]):
                content_type = "ZOOM_SCREEN"
            else:
                content_type = "SLIDE_TEXT"
            by_type[content_type] += 1

            cleaned_text = re.sub(r"<[^>]+>", " ", ocr_text)
            lines = [ln.strip() for ln in cleaned_text.splitlines() if ln.strip()]
            content_summary = lines[0][:180] if lines else ""
            topic_id, topic_name = self._agent25_match_topic(timestamp_sec, cleaned_text, topic_no_vec)

            insertion_priority = {
                "DATA_TABLE": 5,
                "PHOTO": 4,
                "CHART": 4,
                "DOCUMENT": 3,
                "SLIDE_TEXT": 2,
                "ZOOM_SCREEN": 1,
            }.get(content_type, 2)

            render_as = {
                "DATA_TABLE": "html_table",
                "PHOTO": "photo_lightbox",
                "CHART": "chart_embed",
                "DOCUMENT": "document_ref",
                "SLIDE_TEXT": "slide_text",
                "ZOOM_SCREEN": "slide_text",
            }.get(content_type, "slide_text")

            table_html = str(cap.get("table_html", "") or "")
            if not table_html and "<table" in lowered:
                m = re.search(r"(<table[\s\S]*?</table>)", ocr_text, flags=re.IGNORECASE)
                table_html = m.group(1) if m else ""

            manifest.append(
                {
                    "capture_index": capture_index,
                    "timestamp_hms": timestamp_hms,
                    "timestamp_sec": int(timestamp_sec),
                    "image_path": image_path,
                    "content_type": content_type,
                    "content_summary": content_summary,
                    "topic_id": topic_id,
                    "topic_name": topic_name,
                    "insertion_priority": insertion_priority,
                    "caption_th": f"สไลด์เกี่ยวกับ {content_summary or 'ข้อมูลการประชุม'}",
                    "special_pattern": None,
                    "pair_index": None,
                    "render_as": render_as,
                    "table_html": table_html,
                    "ocr_file_size_bytes": file_size,
                }
            )

        return {
            "image_manifest": manifest,
            "statistics": {
                "total": len(cap_chunk),
                "filtered": filtered,
                "by_type": dict(by_type),
                "before_after_pairs": [],
                "data_series": [],
            },
        }

    def _agent25_chunk_recover(
        self,
        cap_chunk: list[dict[str, Any]],
        topic_no_vec: list[dict[str, Any]],
        run_meta: dict[str, Any],
        artifact_dir: str | None,
        *,
        chunk_label: str,
        tag_prefix: str,
    ) -> dict[str, Any]:
        if len(cap_chunk) <= 1:
            self._append_log(
                run_meta,
                artifact_dir,
                "Agent2.5 deterministic fallback",
                chunk=chunk_label,
                captures=len(cap_chunk),
            )
            return self._agent25_chunk_fallback(cap_chunk, topic_no_vec)

        split_at = max(1, len(cap_chunk) // 2)
        parts = [cap_chunk[:split_at], cap_chunk[split_at:]]
        parts = [p for p in parts if p]
        self._append_log(
            run_meta,
            artifact_dir,
            "Agent2.5 split recover",
            chunk=chunk_label,
            subchunks=len(parts),
            split_at=split_at,
        )

        partials: list[dict[str, Any]] = []
        for sidx, sub in enumerate(parts, start=1):
            try:
                partial = self._agent25_call_llm(
                    sub,
                    topic_no_vec,
                    tag=f"{tag_prefix}_sub{sidx}",
                )
            except Exception as sub_exc:
                self._append_log(
                    run_meta,
                    artifact_dir,
                    "Agent2.5 subchunk fallback",
                    chunk=chunk_label,
                    subchunk=f"{sidx}/{len(parts)}",
                    error=str(sub_exc),
                )
                partial = self._agent25_chunk_fallback(sub, topic_no_vec)
            partials.append(partial)
        return merge_partial_image_outputs(partials)

    def node_load_inputs(self, _: WorkflowState) -> WorkflowState:
        out_path = Path(self.cfg.output_html_path)
        ensure_dir(out_path.parent)

        run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        artifact_dir = out_path.parent / "artifacts" / run_id
        if self.cfg.save_intermediate:
            ensure_dir(artifact_dir)

        transcript = load_json(self.cfg.transcript_path)
        raw_config = load_json(self.cfg.config_path) if self.cfg.config_path else {}
        if isinstance(raw_config, dict):
            config_data = raw_config
        elif isinstance(raw_config, list):
            # Allow a compact config file that is only the override list.
            config_data = {"TOPIC_TIME_OVERRIDES": raw_config}
        else:
            config_data = {}
        ocr_data = load_json(self.cfg.ocr_path) if (self.cfg.include_ocr and self.cfg.ocr_path) else {"captures": []}

        segments = transcript.get("segments", [])
        if not isinstance(segments, list) or not segments:
            raise PipelineError("TRANSCRIPT_PATH has no segments")

        captures = ocr_data.get("captures", []) if isinstance(ocr_data.get("captures"), list) else []
        resume_cleaned: dict[str, Any] = {}
        resume_kg: dict[str, Any] = {}
        resume_mismatch_note: dict[str, Any] | None = None
        if self.cfg.resume_artifact_dir:
            resume_dir = Path(self.cfg.resume_artifact_dir).expanduser()
            resume_cleaned_path = resume_dir / "agent1_cleaned.json"
            resume_kg_path = resume_dir / "agent2_kg.json"
            if not resume_cleaned_path.exists():
                raise PipelineError(
                    f"--resume-artifact-dir set but missing file: {resume_cleaned_path}"
                )
            resume_cleaned = load_json(resume_cleaned_path)
            if resume_kg_path.exists():
                resume_kg = load_json(resume_kg_path)

            # Guard against stale resume artifacts that do not cover current input duration.
            transcript_end_sec = 0.0
            for seg in segments:
                if not isinstance(seg, dict):
                    continue
                try:
                    end_sec = float(seg.get("end", seg.get("start", 0)) or 0)
                except Exception:
                    end_sec = 0.0
                if end_sec > transcript_end_sec:
                    transcript_end_sec = end_sec

            resume_timeline = resume_cleaned.get("timeline", []) if isinstance(resume_cleaned, dict) else []
            resume_end_sec = 0.0
            if isinstance(resume_timeline, list):
                for row in resume_timeline:
                    if not isinstance(row, dict):
                        continue
                    try:
                        ts = float(row.get("timestamp_sec", 0) or 0)
                    except Exception:
                        ts = 0.0
                    if ts > resume_end_sec:
                        resume_end_sec = ts

            if transcript_end_sec > 0 and resume_end_sec < (transcript_end_sec * 0.90):
                resume_mismatch_note = {
                    "source": str(resume_dir),
                    "resume_end": sec_to_hms(resume_end_sec),
                    "input_end": sec_to_hms(transcript_end_sec),
                    "resume_timeline": len(resume_timeline) if isinstance(resume_timeline, list) else 0,
                    "input_segments": len(segments),
                }
                resume_cleaned = {}
                resume_kg = {}

        run_meta: dict[str, Any] = {
            "run_id": run_id,
            "started_at": datetime.now().isoformat(),
            "config": {
                "mode": self.cfg.summarize_mode,
                "include_ocr": self.cfg.include_ocr,
                "image_insert_enabled": self.cfg.image_insert_enabled,
                "report_layout_mode": self.cfg.report_layout_mode,
                "image_embed_mode": self.cfg.image_embed_mode,
                "allow_ollama_chat_fallback": self.cfg.allow_ollama_chat_fallback,
                "chat_fallback_provider": self.cfg.chat_fallback_provider,
                "embedding_provider": self.cfg.embedding_provider,
                "agent1_chunk_size": self.cfg.agent1_chunk_size,
                "agent1_chunk_overlap": self.cfg.agent1_chunk_overlap,
                "agent1_subchunk_on_failure": self.cfg.agent1_subchunk_on_failure,
                "agent1_subchunk_size": self.cfg.agent1_subchunk_size,
                "agent1_ocr_max_captures": self.cfg.agent1_ocr_max_captures,
                "agent1_ocr_snippet_chars": self.cfg.agent1_ocr_snippet_chars,
                "agent2_chunk_size": self.cfg.agent2_chunk_size,
                "agent25_chunk_size": self.cfg.agent25_chunk_size,
                "pipeline_max_concurrency": self.cfg.pipeline_max_concurrency,
                "resume_artifact_dir": self.cfg.resume_artifact_dir,
                "save_intermediate": self.cfg.save_intermediate,
            },
            "chunk_stats": {},
            "provider_calls": [],
            "runtime_logs": [],
        }

        self._append_log(
            run_meta,
            str(artifact_dir),
            "Loaded inputs",
            segments=len(segments),
            captures=len(captures),
            mode=self.cfg.summarize_mode,
            concurrency=self.cfg.pipeline_max_concurrency,
        )
        if resume_mismatch_note:
            self._append_log(
                run_meta,
                str(artifact_dir),
                "Resume input ignored",
                reason="timeline_coverage_mismatch",
                source=resume_mismatch_note["source"],
                resume_end=resume_mismatch_note["resume_end"],
                input_end=resume_mismatch_note["input_end"],
                resume_timeline=resume_mismatch_note["resume_timeline"],
                input_segments=resume_mismatch_note["input_segments"],
            )
        self._append_log(
            run_meta,
            str(artifact_dir),
            f"output path : {Path(self.cfg.output_html_path).resolve()}",
        )
        self._append_log(
            run_meta,
            str(artifact_dir),
            f"artifact path : {artifact_dir.resolve()}",
        )
        if resume_cleaned:
            self._append_log(
                run_meta,
                str(artifact_dir),
                "Resume input detected",
                source=self.cfg.resume_artifact_dir,
                timeline=len(resume_cleaned.get("timeline", [])),
                slides=len(resume_cleaned.get("slides", [])),
            )
        if resume_kg:
            topics = resume_kg.get("topics", []) if isinstance(resume_kg.get("topics"), list) else []
            self._append_log(
                run_meta,
                str(artifact_dir),
                "Resume KG detected",
                source=self.cfg.resume_artifact_dir,
                topics=len(topics),
            )

        return {
            "run_id": run_id,
            "artifact_dir": str(artifact_dir),
            "run_meta": run_meta,
            "transcript": transcript,
            "config_data": config_data,
            "ocr_data": ocr_data,
            "segments": segments,
            "captures": captures,
            "resume_cleaned": resume_cleaned,
            "resume_kg": resume_kg,
        }

    def node_agent1(self, state: WorkflowState) -> WorkflowState:
        resume_cleaned = state.get("resume_cleaned", {})
        if isinstance(resume_cleaned, dict) and isinstance(resume_cleaned.get("timeline"), list):
            run_meta = dict(state["run_meta"])
            artifact_dir = state.get("artifact_dir")
            self._append_log(
                run_meta,
                artifact_dir,
                "Agent1 skipped",
                reason="resume_artifact",
                source=self.cfg.resume_artifact_dir,
            )
            self._save_json_if_enabled(state, "agent1_cleaned.json", resume_cleaned)
            return {"cleaned": resume_cleaned, "run_meta": run_meta}

        segments = state["segments"]
        captures = state.get("captures", [])
        config_data = state.get("config_data", {})
        artifact_dir = state.get("artifact_dir")

        seg_chunks = chunked(
            segments,
            max(1, self.cfg.agent1_chunk_size),
            overlap=max(0, self.cfg.agent1_chunk_overlap),
        )
        run_meta = dict(state["run_meta"])
        run_meta.setdefault("chunk_stats", {})["agent1_segment_chunks"] = len(seg_chunks)
        self._append_log(
            run_meta,
            artifact_dir,
            "Agent1 map start",
            chunks=len(seg_chunks),
            segments=len(segments),
            chunk_size=self.cfg.agent1_chunk_size,
            overlap=self.cfg.agent1_chunk_overlap,
            mode="split_transcript_llm_ocr_llm",
            ocr_payload="compact_snippet",
            ocr_max_captures=self.cfg.agent1_ocr_max_captures,
            ocr_snippet_chars=self.cfg.agent1_ocr_snippet_chars,
        )

        outputs: list[dict[str, Any]] = []
        for idx, seg_chunk in enumerate(seg_chunks, start=1):
            start_sec = float(seg_chunk[0].get("start", 0) or 0)
            end_sec = float(seg_chunk[-1].get("end", start_sec) or start_sec)
            ocr_subset = self._select_agent1_ocr_subset(captures, start_sec, end_sec, window_sec=120.0)
            self._append_log(
                run_meta,
                artifact_dir,
                "Agent1 chunk",
                chunk=f"{idx}/{len(seg_chunks)}",
                seg=len(seg_chunk),
                ocr=len(ocr_subset),
                start=sec_to_hms(start_sec),
                end=sec_to_hms(end_sec),
            )

            # Transcript path: LLM first (OCR removed), deterministic fallback on failure.
            self._append_log(
                run_meta,
                artifact_dir,
                "Agent1 transcript-only call",
                chunk=f"{idx}/{len(seg_chunks)}",
                ocr_payload=0,
            )
            try:
                out = self._agent1_call_llm_transcript_only(
                    seg_chunk=seg_chunk,
                    config_data=config_data,
                    tag=f"agent1_transcript_chunk_{idx}",
                )
                self._append_log(
                    run_meta,
                    artifact_dir,
                    "Agent1 transcript-only done",
                    chunk=f"{idx}/{len(seg_chunks)}",
                    timeline=len(out.get("timeline", [])) if isinstance(out, dict) else 0,
                )
            except Exception as exc:
                self._append_log(
                    run_meta,
                    artifact_dir,
                    "Agent1 transcript-only fallback",
                    chunk=f"{idx}/{len(seg_chunks)}",
                    error=str(exc),
                )
                out = self._agent1_transcript_split_recover(
                    seg_chunk=seg_chunk,
                    config_data=config_data,
                    run_meta=run_meta,
                    artifact_dir=artifact_dir,
                    parent_chunk_idx=idx,
                    parent_chunk_total=len(seg_chunks),
                )

            # OCR path: optional LLM enrichment on OCR-only payload.
            if ocr_subset:
                self._append_log(
                    run_meta,
                    artifact_dir,
                    "Agent1 ocr-only call",
                    chunk=f"{idx}/{len(seg_chunks)}",
                    ocr=len(ocr_subset),
                )
                try:
                    ocr_out = self._agent1_call_llm_ocr_only(
                        ocr_subset=ocr_subset,
                        config_data=config_data,
                        tag=f"agent1_ocr_chunk_{idx}",
                    )
                    slides = ocr_out.get("slides", [])
                    meeting_meta = ocr_out.get("meeting_meta", {})
                    if isinstance(slides, list) and slides:
                        out["slides"] = slides
                    self._append_log(
                        run_meta,
                        artifact_dir,
                        "Agent1 ocr-only done",
                        chunk=f"{idx}/{len(seg_chunks)}",
                        slides=len(slides) if isinstance(slides, list) else 0,
                    )
                    if isinstance(meeting_meta, dict) and meeting_meta:
                        base_meta = out.get("meeting_meta", {})
                        if not isinstance(base_meta, dict):
                            base_meta = {}
                        merged_meta = dict(meeting_meta)
                        merged_meta.update({k: v for k, v in base_meta.items() if v not in ("", [], None)})
                        out["meeting_meta"] = merged_meta
                except Exception as exc:
                    self._append_log(
                        run_meta,
                        artifact_dir,
                        "Agent1 OCR-only fallback",
                        chunk=f"{idx}/{len(seg_chunks)}",
                        error=str(exc),
                    )

            assert isinstance(out, dict)
            outputs.append(out)
            self._append_log(
                run_meta,
                artifact_dir,
                "Agent1 chunk done",
                chunk=f"{idx}/{len(seg_chunks)}",
                timeline=len(out.get("timeline", [])),
                slides=len(out.get("slides", [])),
            )

        self._append_log(run_meta, artifact_dir, "Agent1 reduce start", partials=len(outputs))
        cleaned = reduce_agent1_maps(outputs, config_data)
        self._append_log(
            run_meta,
            artifact_dir,
            "Agent1 reduce done",
            timeline=len(cleaned.get("timeline", [])),
            slides=len(cleaned.get("slides", [])),
        )
        self._save_json_if_enabled(state, "agent1_cleaned.json", cleaned)

        return {"cleaned": cleaned, "run_meta": run_meta}

    def node_agent2(self, state: WorkflowState) -> WorkflowState:
        resume_kg = state.get("resume_kg", {})
        if isinstance(resume_kg, dict) and isinstance(resume_kg.get("topics"), list):
            topics = resume_kg.get("topics", [])
            run_meta = dict(state["run_meta"])
            artifact_dir = state.get("artifact_dir")
            if topics:
                topic_texts = [build_topic_text(t) for t in topics if isinstance(t, dict)]
                topic_vecs: list[list[float]] = []
                if topic_texts:
                    try:
                        topic_vecs = self.llm.embed(topic_texts)
                    except Exception as exc:
                        self._append_log(
                            run_meta,
                            artifact_dir,
                            "Agent2 embedding failed",
                            error=str(exc),
                            action="continue_without_vectors",
                        )
                for i, topic in enumerate(topics):
                    if isinstance(topic, dict):
                        topic["_vec"] = topic_vecs[i] if i < len(topic_vecs) else []
                kg = resume_kg
                self._append_log(
                    run_meta,
                    artifact_dir,
                    "Agent2 skipped",
                    reason="resume_artifact",
                    source=self.cfg.resume_artifact_dir,
                    topics=len(topics),
                )
                self._save_json_if_enabled(state, "agent2_kg.json", sanitize_kg_for_output(kg))
                return {"kg": kg, "topics": topics, "run_meta": run_meta}
            self._append_log(
                run_meta,
                artifact_dir,
                "Resume KG ignored",
                reason="no_topics",
                source=self.cfg.resume_artifact_dir,
            )

        cleaned = state["cleaned"]
        timeline = cleaned.get("timeline", [])
        if not isinstance(timeline, list) or not timeline:
            raise PipelineError("Agent1 output has empty timeline")
        artifact_dir = state.get("artifact_dir")

        timeline_chunks = chunked(timeline, max(1, self.cfg.agent2_chunk_size), overlap=0)
        run_meta = dict(state["run_meta"])
        run_meta.setdefault("chunk_stats", {})["agent2_timeline_chunks"] = len(timeline_chunks)
        self._append_log(
            run_meta,
            artifact_dir,
            "Agent2 map start",
            chunks=len(timeline_chunks),
            timeline=len(timeline),
            chunk_size=self.cfg.agent2_chunk_size,
            workers=self._effective_workers(len(timeline_chunks)),
        )

        slides = cleaned.get("slides", []) if isinstance(cleaned.get("slides"), list) else []
        partial_kgs: list[dict[str, Any]] = []
        chunk_inputs: list[tuple[int, list[dict[str, Any]], list[dict[str, Any]]]] = []
        total_chunks = len(timeline_chunks)
        for idx, tl_chunk in enumerate(timeline_chunks, start=1):
            c_start = float(tl_chunk[0].get("timestamp_sec", 0) or 0)
            c_end = float(tl_chunk[-1].get("timestamp_sec", c_start) or c_start)
            slides_subset = [
                s
                for s in slides
                if (c_start - 120.0)
                <= hms_to_sec(str(s.get("timestamp_hms", "00:00:00")))
                <= (c_end + 120.0)
            ]
            self._append_log(
                run_meta,
                artifact_dir,
                "Agent2 chunk",
                chunk=f"{idx}/{len(timeline_chunks)}",
                rows=len(tl_chunk),
                slides=len(slides_subset),
                start=sec_to_hms(c_start),
                end=sec_to_hms(c_end),
            )
            chunk_inputs.append((idx, tl_chunk, slides_subset))

        def run_one_chunk(item: tuple[int, list[dict[str, Any]], list[dict[str, Any]]]) -> tuple[int, dict[str, Any]]:
            idx, tl_chunk, slides_subset = item
            data = {
                "meeting_meta": cleaned.get("meeting_meta", {}),
                "timeline": tl_chunk,
                "slides": slides_subset,
            }
            user = fill_template(AGENT2_USR, DATA=json.dumps(data, ensure_ascii=False))
            out = self.llm.call(
                AGENT2_SYS,
                user,
                json_mode=True,
                # Allow partial outputs and sanitize in-code to avoid losing whole chunks.
                required_keys=["entities"],
                tag=f"agent2_map_chunk_{idx}",
            )
            assert isinstance(out, dict)
            entities = out.get("entities", {})
            if not isinstance(entities, dict):
                entities = {}
            for key in ["people", "projects", "equipment", "financials", "issues", "decisions", "action_items"]:
                if not isinstance(entities.get(key), list):
                    entities[key] = []
            out["entities"] = entities

            raw_topics = out.get("topics", [])
            if not isinstance(raw_topics, list):
                raw_topics = []
            kept_topics: list[dict[str, Any]] = []
            for raw in raw_topics:
                if not isinstance(raw, dict):
                    continue
                has_signal = (
                    bool(str(raw.get("name", "") or "").strip())
                    or bool(str(raw.get("title", "") or "").strip())
                    or bool(str(raw.get("start_timestamp", "") or "").strip())
                    or bool(str(raw.get("end_timestamp", "") or "").strip())
                    or bool([x for x in (raw.get("summary_points", []) if isinstance(raw.get("summary_points"), list) else []) if str(x).strip()])
                    or bool([x for x in (raw.get("decisions", []) if isinstance(raw.get("decisions"), list) else []) if str(x).strip()])
                    or bool([x for x in (raw.get("action_items", []) if isinstance(raw.get("action_items"), list) else []) if str(x).strip()])
                )
                if not has_signal:
                    continue
                norm = self._normalize_topic(raw, len(kept_topics) + 1)
                if norm:
                    kept_topics.append(norm)
            out["topics"] = kept_topics
            return idx, out

        workers = self._effective_workers(len(chunk_inputs))
        if workers == 1:
            for idx, tl_chunk, slides_subset in chunk_inputs:
                try:
                    _, out = run_one_chunk((idx, tl_chunk, slides_subset))
                except Exception as exc:
                    self._append_log(
                        run_meta,
                        artifact_dir,
                        "Agent2 chunk failed",
                        chunk=f"{idx}/{total_chunks}",
                        error=str(exc),
                        action="deterministic_fallback",
                    )
                    out = self._agent2_chunk_fallback(tl_chunk, slides_subset, idx)
                partial_kgs.append(out)
                self._append_log(
                    run_meta,
                    artifact_dir,
                    "Agent2 chunk done",
                    chunk=f"{idx}/{total_chunks}",
                    topics=len(out.get("topics", [])),
                )
        else:
            results: dict[int, dict[str, Any]] = {}
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(run_one_chunk, item): item for item in chunk_inputs}
                for future in as_completed(futures):
                    idx, tl_chunk, slides_subset = futures[future]
                    try:
                        _, out = future.result()
                    except Exception as exc:
                        self._append_log(
                            run_meta,
                            artifact_dir,
                            "Agent2 chunk failed",
                            chunk=f"{idx}/{total_chunks}",
                            error=str(exc),
                            action="deterministic_fallback",
                        )
                        out = self._agent2_chunk_fallback(tl_chunk, slides_subset, idx)
                    results[idx] = out
                    self._append_log(
                        run_meta,
                        artifact_dir,
                        "Agent2 chunk done",
                        chunk=f"{idx}/{total_chunks}",
                        topics=len(out.get("topics", [])),
                    )
            for idx in sorted(results):
                partial_kgs.append(results[idx])

        self._append_log(run_meta, artifact_dir, "Agent2 reduce start", partials=len(partial_kgs))
        if not partial_kgs:
            kg = self._agent2_deterministic_fallback([], cleaned)
        elif len(partial_kgs) == 1:
            kg = partial_kgs[0]
        else:
            reduce_user = fill_template(
                AGENT2_REDUCE_USR,
                PARTIAL_KGS=json.dumps(partial_kgs, ensure_ascii=False),
            )
            try:
                reduced = self.llm.call(
                    AGENT2_REDUCE_SYS,
                    reduce_user,
                    json_mode=True,
                    required_keys=["entities", "topics"],
                    tag="agent2_reduce",
                )
                assert isinstance(reduced, dict)
                kg = reduced
            except Exception as exc:
                self._append_log(
                    run_meta,
                    artifact_dir,
                    "Agent2 reduce failed",
                    error=str(exc),
                    action="deterministic_fallback",
                )
                kg = self._agent2_deterministic_fallback(partial_kgs, cleaned)

        topics = kg.get("topics", []) if isinstance(kg.get("topics"), list) else []
        if not topics:
            self._append_log(
                run_meta,
                artifact_dir,
                "Agent2 reduce empty topics",
                action="deterministic_fallback",
            )
            kg = self._agent2_deterministic_fallback(partial_kgs, cleaned)
            topics = kg.get("topics", []) if isinstance(kg.get("topics"), list) else []

        if not topics:
            raise PipelineError("Agent2 produced no topics (including fallback)")

        topic_texts = [build_topic_text(t) for t in topics]
        topic_vecs: list[list[float]] = []
        if topic_texts:
            try:
                topic_vecs = self.llm.embed(topic_texts)
            except Exception as exc:
                self._append_log(
                    run_meta,
                    artifact_dir,
                    "Agent2 embedding failed",
                    error=str(exc),
                    action="continue_without_vectors",
                )
        for i, topic in enumerate(topics):
            topic["_vec"] = topic_vecs[i] if i < len(topic_vecs) else []
        self._append_log(
            run_meta,
            artifact_dir,
            "Agent2 reduce done",
            topics=len(topics),
            decisions=len(kg.get("entities", {}).get("decisions", [])),
            actions=len(kg.get("entities", {}).get("action_items", [])),
        )

        self._save_json_if_enabled(state, "agent2_kg.json", sanitize_kg_for_output(kg))
        return {"kg": kg, "topics": topics, "run_meta": run_meta}

    def route_after_agent2(self, state: WorkflowState) -> str:
        config_data = state.get("config_data", {})
        agenda_text = str(config_data.get("AGENDA_TEXT", "") or "")
        run_meta = state.get("run_meta", {})
        artifact_dir = state.get("artifact_dir")
        if self.cfg.summarize_mode == "agenda" and agenda_text.strip():
            self._append_log(run_meta, artifact_dir, "Routing", next="agent3a", mode="agenda")
            return "agent3a"
        self._append_log(run_meta, artifact_dir, "Routing", next="agent3b", mode="auto")
        return "agent3b"

    def node_agent3a(self, state: WorkflowState) -> WorkflowState:
        run_meta = dict(state["run_meta"])
        artifact_dir = state.get("artifact_dir")
        config_data = state.get("config_data", {})
        agenda_text = str(config_data.get("AGENDA_TEXT", "") or "")
        topics = state["topics"]

        agenda_lines = [x.strip() for x in agenda_text.splitlines() if x.strip()]
        agenda_vecs: list[list[float]] = []
        if agenda_lines:
            try:
                agenda_vecs = self.llm.embed(agenda_lines)
            except Exception as exc:
                self._append_log(
                    run_meta,
                    artifact_dir,
                    "Agent3A embedding failed",
                    error=str(exc),
                    action="continue_without_vectors",
                )
        semantic_hints = []
        self._append_log(
            run_meta,
            artifact_dir,
            "Agent3A start",
            agenda_lines=len(agenda_lines),
            topics=len(topics),
        )

        import re
        def extract_keywords(text: str) -> list[str]:
            # Remove common generic stop words
            stopwords = ["สรุป", "รายงาน", "ความเสียหาย", "สูญหาย", "ของ", "ประจำ", "เดือน", "ปี", "ทรัพย์สิน", "และ"]
            words = re.findall(r'[a-zA-Z]+|[\u0E00-\u0E7F]+', text)
            return [w for w in words if len(w) > 2 and w not in stopwords]

        def find_transcript_hints(line: str, timeline: list) -> list[str]:
            if not timeline:
                return []
            high_signal = ["บางไทร", "โกดัง", "protection", "โปรดิกชั่น", "โปรเทคชั่น", "defect", "ดีเฟค", "ห้องพัก", "ประมูล", "สูญหาย", "คนงาน", "ต่างด้าว"]
            line_lower = line.lower()
            active_signals = [w for w in high_signal if w in line_lower]
            
            stopwords = ["สรุป", "รายงาน", "ความเสียหาย", "ของ", "ประจำ", "เดือน", "ปี", "ทรัพย์สิน", "และ", "งาน", "ผล", "การ", "ติดตาม", "แจ้ง", "เรื่อง", "ที่", "ใน", "ให้"]
            words = re.findall(r'[a-zA-Z]+|[\u0E00-\u0E7F]+', line_lower)
            generic_keywords = [w for w in words if len(w) > 3 and w not in stopwords]

            found_secs = []
            for item in timeline:
                text = str(item.get("text", "")).lower()
                sec = float(item.get("timestamp_sec", 0))
                if active_signals and any(sig in text for sig in active_signals):
                    found_secs.append(sec)
                elif len(generic_keywords) >= 2:
                    if sum(1 for kw in generic_keywords if kw in text) >= 2:
                        found_secs.append(sec)
            
            if not found_secs:
                return []
                
            found_secs.sort()
            clusters = []
            curr_cluster = [found_secs[0]]
            for s in found_secs[1:]:
                if s - curr_cluster[-1] <= 300: # 5 minutes gap max
                    curr_cluster.append(s)
                else:
                    clusters.append(curr_cluster)
                    curr_cluster = [s]
            clusters.append(curr_cluster)
            
            hint_ranges = []
            for c in clusters:
                start_hms = sec_to_hms(c[0])
                end_hms = sec_to_hms(c[-1] + 180) # Add a 3 minute buffer to the last hit
                if start_hms == end_hms:
                    hint_ranges.append(start_hms)
                else:
                    hint_ranges.append(f"{start_hms} to {end_hms}")
            return hint_ranges

        timeline_data = state.get("cleaned", {}).get("timeline", [])

        for idx_line, line in enumerate(agenda_lines):
            avec = agenda_vecs[idx_line] if idx_line < len(agenda_vecs) else []
            scores = []
            for t in topics:
                vec = t.get("_vec", [])
                score = cosine(avec, vec) if (isinstance(avec, list) and isinstance(vec, list) and avec and vec) else 0.0
                scores.append((score, str(t.get("id", "")), str(t.get("name", "")), str(t.get("description", ""))))
            
            top3 = sorted(scores, key=lambda x: x[0], reverse=True)[:3]
            
            # --- Keyword & Transcript Matching ---
            kw_matches = []
            keywords = extract_keywords(line)
            if keywords:
                for t in topics:
                    t_text = str(t.get("name", "")) + " " + str(t.get("description", ""))
                    if any(kw.lower() in t_text.lower() for kw in keywords):
                        kw_matches.append({"topic_id": t.get("id", ""), "topic_name": t.get("name", "")})
            
            transcript_times = find_transcript_hints(line, timeline_data)
            
            semantic_hints.append(
                {
                    "agenda_line": line,
                    "extracted_keywords_to_look_for": keywords,
                    "keyword_matches_found_in_kg": kw_matches[:3],
                    "transcript_timestamps_where_discussed": transcript_times,
                    "semantic_best_topic": top3[0][1] if top3 else "",
                    "semantic_score": round(top3[0][0], 3) if top3 else 0.0,
                    "semantic_alternatives": [
                        {"topic_id": t[1], "topic_name": t[2], "score": round(t[0], 3)}
                        for t in top3[1:]
                    ],
                }
            )

        user = fill_template(
            AGENT3A_USR,
            AGENDA=agenda_text,
            KG_TOPICS=json.dumps(sanitize_kg_for_output({"topics": topics})["topics"], ensure_ascii=False),
            SEMANTIC_HINTS=json.dumps(semantic_hints, ensure_ascii=False),
        )
        try:
            topic_map = self.llm.call(
                AGENT3A_SYS,
                user,
                json_mode=True,
                required_keys=["agenda_mapping", "coverage_stats"],
                tag="agent3a",
            )
            assert isinstance(topic_map, dict)
        except Exception as exc:
            self._append_log(
                run_meta,
                artifact_dir,
                "Agent3A failed",
                error=str(exc),
                action="deterministic_fallback",
            )
            topic_map = self._agent3a_fallback_from_hints(agenda_lines, semantic_hints, topics)
        self._append_log(
            run_meta,
            artifact_dir,
            "Agent3A done",
            mapped=len(topic_map.get("agenda_mapping", [])),
            discussed=topic_map.get("coverage_stats", {}).get("discussed", 0),
        )

        # --- Post‑process: fix fabricated time_ranges using KG topic timestamps ---
        # Agent3A frequently fabricates time_ranges that extend beyond the actual
        # meeting duration, causing Agent4 to get empty timeline snippets.
        # Replace with KG topic's real timestamps when available.
        topic_time_lookup: dict[str, dict[str, str]] = {}
        for t in topics:
            if isinstance(t, dict):
                tid = str(t.get("id", ""))
                if tid:
                    topic_time_lookup[tid] = {
                        "start": str(t.get("start_timestamp", "00:00:00")),
                        "end": str(t.get("end_timestamp", "00:00:00")),
                    }

        # Get actual meeting end time from timeline
        timeline = state.get("cleaned", {}).get("timeline", [])
        meeting_end_sec = 0
        if timeline:
            last_entry = timeline[-1] if isinstance(timeline[-1], dict) else {}
            meeting_end_sec = int(float(last_entry.get("timestamp_sec", 0) or 0))

        # Build lookup to check if agendas had explicit transcript timestamps
        agenda_has_hints = {}
        for hint in semantic_hints:
            line = str(hint.get("agenda_line", ""))
            agenda_has_hints[line] = len(hint.get("transcript_timestamps_where_discussed", [])) > 0

        time_fixes = 0
        for item in topic_map.get("agenda_mapping", []):
            if not isinstance(item, dict):
                continue
            mapped = item.get("mapped_topics", [])
            if not isinstance(mapped, list) or not mapped:
                continue
            tr = item.get("time_range", {})
            if not isinstance(tr, dict):
                continue
            start_str = str(tr.get("start", "00:00:00"))
            end_str = str(tr.get("end", "00:00:00"))
            start_sec = hms_to_sec(start_str)
            end_sec = hms_to_sec(end_str)

            agenda_title = str(item.get("agenda_title", ""))
            had_hint = False
            for line, has_hint in agenda_has_hints.items():
                if agenda_title in line or line in agenda_title:
                    had_hint = has_hint
                    break

            # If the time_range is beyond the meeting end or doesn't overlap
            # with the KG topic, replace it with the KG topic's timestamps.
            topic_id = str(mapped[0])
            kg_tr = topic_time_lookup.get(topic_id, {})
            if kg_tr:
                kg_start = hms_to_sec(kg_tr.get("start", "00:00:00"))
                kg_end = hms_to_sec(kg_tr.get("end", "00:00:00"))
                # Only fix if the time_range is completely fabricated beyond
                # the physical end of the meeting. If it's within the meeting,
                # trust the LLM's sequence logic over the KG topic ONLY IF it actually
                # received a transcript hint. If no hint and 0% overlap, it's a hallucination.
                needs_fix = (
                    start_sec > meeting_end_sec
                    or end_sec > meeting_end_sec + 300
                )
                if not had_hint and (end_sec <= kg_start or start_sec >= kg_end):
                    needs_fix = True
                if needs_fix:
                    item["time_range"] = kg_tr
                    item["_time_range_fixed"] = True
                    time_fixes += 1
        if time_fixes:
            self._append_log(
                run_meta,
                artifact_dir,
                "Agent3A time_range post-fix",
                fixed=time_fixes,
                total=len(topic_map.get("agenda_mapping", [])),
            )

        override_applied = self._apply_topic_time_overrides(topic_map, config_data)
        if override_applied:
            self._append_log(
                run_meta,
                artifact_dir,
                "Agent3A topic_time_overrides applied",
                count=override_applied,
            )

        self._save_json_if_enabled(state, "agent3_topic_map.json", topic_map)
        return {"topic_map": topic_map, "run_meta": run_meta}

    def node_agent3b(self, state: WorkflowState) -> WorkflowState:
        run_meta = dict(state["run_meta"])
        artifact_dir = state.get("artifact_dir")
        config_data = state.get("config_data", {})
        kg = state["kg"]
        topics = state["topics"]
        timeline = state["cleaned"].get("timeline", [])
        timeline_sample = self._sample_timeline_for_agent3b(timeline, max_items=300)
        self._append_log(
            run_meta,
            artifact_dir,
            "Agent3B start",
            timeline_total=len(timeline),
            timeline_sample=len(timeline_sample),
        )

        user = fill_template(
            AGENT3B_USR,
            KG=json.dumps(sanitize_kg_for_output(kg), ensure_ascii=False),
            TIMELINE=json.dumps(timeline_sample, ensure_ascii=False),
        )
        try:
            topic_map = self.llm.call(
                AGENT3B_SYS,
                user,
                json_mode=True,
                required_keys=["extracted_topics", "topic_flow"],
                tag="agent3b",
            )
            assert isinstance(topic_map, dict)
        except Exception as exc:
            self._append_log(
                run_meta,
                artifact_dir,
                "Agent3B failed",
                error=str(exc),
                action="fallback_from_kg",
            )
            topic_map = self._agent3b_fallback_from_kg(topics)
        extracted = topic_map.get("extracted_topics", [])
        if not isinstance(extracted, list):
            extracted = []
        coverage = self._topic_coverage_ratio(extracted, timeline)
        if len(extracted) < 8 or coverage < 0.6:
            self._append_log(
                run_meta,
                artifact_dir,
                "Agent3B fallback from KG",
                reason="low_topic_count_or_coverage",
                extracted=len(extracted),
                coverage=f"{coverage:.2f}",
            )
            topic_map = self._agent3b_fallback_from_kg(topics)
            extracted = topic_map.get("extracted_topics", []) if isinstance(topic_map.get("extracted_topics"), list) else []

        override_applied = self._apply_topic_time_overrides(topic_map, config_data)
        if override_applied:
            self._append_log(
                run_meta,
                artifact_dir,
                "Agent3B topic_time_overrides applied",
                count=override_applied,
            )
            extracted = topic_map.get("extracted_topics", []) if isinstance(topic_map.get("extracted_topics"), list) else []

        self._append_log(
            run_meta,
            artifact_dir,
            "Agent3B done",
            extracted=len(extracted),
            coverage=f"{self._topic_coverage_ratio(extracted, timeline):.2f}",
        )

        self._save_json_if_enabled(state, "agent3_topic_map.json", topic_map)
        return {"topic_map": topic_map, "run_meta": run_meta}

    def node_agent25(self, state: WorkflowState) -> WorkflowState:
        captures = state.get("captures", [])
        topics = state["topics"]
        run_meta = dict(state["run_meta"])
        artifact_dir = state.get("artifact_dir")

        image_by_topic: dict[str, list[dict[str, Any]]] = {}
        image_manifest_output: dict[str, Any] = {"image_manifest": [], "statistics": {}}

        if not (self.cfg.image_insert_enabled and self.cfg.include_ocr and captures):
            self._append_log(run_meta, artifact_dir, "Agent2.5 skipped", reason="image_insert_disabled_or_no_ocr")
            return {
                "image_by_topic": image_by_topic,
                "image_manifest_output": image_manifest_output,
                "run_meta": run_meta,
            }

        capture_chunks = chunked(captures, max(1, self.cfg.agent25_chunk_size), overlap=0)
        run_meta.setdefault("chunk_stats", {})["agent25_capture_chunks"] = len(capture_chunks)
        self._append_log(
            run_meta,
            artifact_dir,
            "Agent2.5 map start",
            chunks=len(capture_chunks),
            captures=len(captures),
            chunk_size=self.cfg.agent25_chunk_size,
            workers=self._effective_workers(len(capture_chunks)),
        )

        partial_images: list[dict[str, Any]] = []
        topic_no_vec = sanitize_kg_for_output({"topics": topics}).get("topics", [])
        chunk_inputs: list[tuple[int, list[dict[str, Any]]]] = []
        total_chunks = len(capture_chunks)
        for idx, cap_chunk in enumerate(capture_chunks, start=1):
            c_start = float(cap_chunk[0].get("timestamp_sec", 0) or 0)
            c_end = float(cap_chunk[-1].get("timestamp_sec", c_start) or c_start)
            self._append_log(
                run_meta,
                artifact_dir,
                "Agent2.5 chunk",
                chunk=f"{idx}/{len(capture_chunks)}",
                captures=len(cap_chunk),
                start=sec_to_hms(c_start),
                end=sec_to_hms(c_end),
            )
            chunk_inputs.append((idx, cap_chunk))

        def run_one_chunk(item: tuple[int, list[dict[str, Any]]]) -> tuple[int, dict[str, Any]]:
            idx, cap_chunk = item
            out = self._agent25_call_llm(cap_chunk, topic_no_vec, tag=f"agent25_map_chunk_{idx}")
            return idx, out

        workers = self._effective_workers(len(chunk_inputs))
        if workers == 1:
            for idx, cap_chunk in chunk_inputs:
                try:
                    _, out = run_one_chunk((idx, cap_chunk))
                except Exception as exc:
                    chunk_label = f"{idx}/{total_chunks}"
                    self._append_log(
                        run_meta,
                        artifact_dir,
                        "Agent2.5 chunk failed",
                        chunk=chunk_label,
                        error=str(exc),
                    )
                    try:
                        out = self._agent25_chunk_recover(
                            cap_chunk,
                            topic_no_vec,
                            run_meta,
                            artifact_dir,
                            chunk_label=chunk_label,
                            tag_prefix=f"agent25_map_chunk_{idx}",
                        )
                    except Exception as recover_exc:
                        self._append_log(
                            run_meta,
                            artifact_dir,
                            "Agent2.5 recover failed",
                            chunk=chunk_label,
                            error=str(recover_exc),
                            action="deterministic_fallback",
                        )
                        out = self._agent25_chunk_fallback(cap_chunk, topic_no_vec)
                partial_images.append(out)
                self._append_log(
                    run_meta,
                    artifact_dir,
                    "Agent2.5 chunk done",
                    chunk=f"{idx}/{total_chunks}",
                    manifest=len(out.get("image_manifest", [])),
                )
        else:
            results: dict[int, dict[str, Any]] = {}
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(run_one_chunk, item): item for item in chunk_inputs}
                for future in as_completed(futures):
                    idx, cap_chunk = futures[future]
                    chunk_label = f"{idx}/{total_chunks}"
                    try:
                        _, out = future.result()
                    except Exception as exc:
                        self._append_log(
                            run_meta,
                            artifact_dir,
                            "Agent2.5 chunk failed",
                            chunk=chunk_label,
                            error=str(exc),
                        )
                        try:
                            out = self._agent25_chunk_recover(
                                cap_chunk,
                                topic_no_vec,
                                run_meta,
                                artifact_dir,
                                chunk_label=chunk_label,
                                tag_prefix=f"agent25_map_chunk_{idx}",
                            )
                        except Exception as recover_exc:
                            self._append_log(
                                run_meta,
                                artifact_dir,
                                "Agent2.5 recover failed",
                                chunk=chunk_label,
                                error=str(recover_exc),
                                action="deterministic_fallback",
                            )
                            out = self._agent25_chunk_fallback(cap_chunk, topic_no_vec)
                    results[idx] = out
                    self._append_log(
                        run_meta,
                        artifact_dir,
                        "Agent2.5 chunk done",
                        chunk=chunk_label,
                        manifest=len(out.get("image_manifest", [])),
                    )
            for idx in sorted(results):
                partial_images.append(results[idx])

        self._append_log(run_meta, artifact_dir, "Agent2.5 reduce start", partials=len(partial_images))
        if len(partial_images) == 1:
            merged_images = partial_images[0]
        else:
            reduce_user = fill_template(
                AGENT25_REDUCE_USR,
                PARTIAL_OUTPUTS=json.dumps(partial_images, ensure_ascii=False),
            )
            try:
                reduced = self.llm.call(
                    AGENT25_REDUCE_SYS,
                    reduce_user,
                    json_mode=True,
                    required_keys=["image_manifest", "statistics"],
                    tag="agent25_reduce",
                )
                assert isinstance(reduced, dict)
                merged_images = reduced
            except Exception:
                merged_images = merge_partial_image_outputs(partial_images)

        merged_images = merge_partial_image_outputs([merged_images])
        manifest = merged_images.get("image_manifest", [])
        if not isinstance(manifest, list):
            manifest = []

        topic_vec_map = {
            str(t.get("id", "")): t.get("_vec", [])
            for t in topics
            if isinstance(t, dict) and str(t.get("id", ""))
        }
        topic_name_map = {
            str(t.get("id", "")): str(t.get("name", "") or "")
            for t in topics
            if isinstance(t, dict) and str(t.get("id", ""))
        }

        cap_texts = [str(c.get("ocr_text", "") or "")[:800] for c in captures]
        cap_vecs = self.llm.embed(cap_texts)
        capture_by_index = {
            int(c.get("capture_index", 0) or 0): c for c in captures if int(c.get("capture_index", 0) or 0) > 0
        }
        replaced_image_paths = 0
        resolved_image_paths = 0

        for item in manifest:
            capture_index = int(item.get("capture_index", 0) or 0)
            render_as = str(item.get("render_as", "") or "")
            special = str(item.get("special_pattern", "") or "")
            is_before_after = special == "BEFORE_AFTER" or render_as == "before_after"
            if 1 <= capture_index <= len(cap_vecs):
                cap_vec = cap_vecs[capture_index - 1]
                best_score = -1.0
                best_id = ""
                for tid, tvec in topic_vec_map.items():
                    if isinstance(tvec, list) and tvec:
                        score = cosine(cap_vec, tvec)
                        if score > best_score:
                            best_score = score
                            best_id = tid
                if best_id and best_score > 0.65:
                    item["topic_id"] = best_id
                    item["topic_name"] = topic_name_map.get(best_id, item.get("topic_name", ""))

            # Do not trust LLM-generated image_path. Always normalize from OCR capture index when available.
            raw_path = str(item.get("image_path", "") or "")
            source_cap = capture_by_index.get(capture_index)
            if source_cap:
                source_path = str(source_cap.get("image_path", "") or "")
                if source_path:
                    if raw_path and raw_path != source_path:
                        item["llm_image_path"] = raw_path
                    if raw_path != source_path:
                        replaced_image_paths += 1
                    item["image_path"] = source_path
                    raw_path = source_path

            if is_before_after and raw_path:
                item["before_image_path"] = raw_path

            resolved = resolve_image_path(raw_path, self.cfg.image_base_dir, self.cfg.ocr_path)
            if resolved:
                item["resolved_image_path"] = str(resolved)
                resolved_image_paths += 1

            if self.cfg.image_embed_mode == "base64" and resolved and render_as in {"photo_lightbox", "before_after"}:
                item["image_base64"] = image_to_base64_data_uri(resolved)

            if is_before_after:
                pair_index = int(item.get("pair_index", 0) or 0)
                if pair_index > 0 and pair_index in capture_by_index:
                    pair_cap = capture_by_index[pair_index]
                    pair_source_path = str(pair_cap.get("image_path", "") or "")
                    if pair_source_path:
                        item["after_image_path"] = pair_source_path
                    pair_resolved = resolve_image_path(
                        pair_source_path,
                        self.cfg.image_base_dir,
                        self.cfg.ocr_path,
                    )
                    if resolved and self.cfg.image_embed_mode == "base64":
                        item["before_base64"] = image_to_base64_data_uri(resolved)
                    if pair_resolved and self.cfg.image_embed_mode == "base64":
                        item["after_base64"] = image_to_base64_data_uri(pair_resolved)

        self._append_log(
            run_meta,
            artifact_dir,
            "Agent2.5 image paths normalized",
            replaced=replaced_image_paths,
            resolved=resolved_image_paths,
        )

        image_by_topic = group_manifest_by_topic(
            manifest,
            max_per_topic=self.cfg.image_max_per_topic,
            min_file_size_kb=self.cfg.image_min_file_size_kb,
        )
        image_manifest_output = {
            "image_manifest": manifest,
            "statistics": merged_images.get("statistics", {}),
            "image_by_topic": image_by_topic,
        }
        self._append_log(
            run_meta,
            artifact_dir,
            "Agent2.5 done",
            manifest=len(manifest),
            topics=len(image_by_topic),
        )

        self._save_json_if_enabled(state, "agent25_image_manifest.json", image_manifest_output)
        return {
            "image_by_topic": image_by_topic,
            "image_manifest_output": image_manifest_output,
            "run_meta": run_meta,
        }

    def _filter_kg_for_time_range(
        self,
        kg: dict[str, Any],
        start_sec: int,
        end_sec: int,
        margin_sec: int = 60,
    ) -> dict[str, Any]:
        """Return a copy of *kg* containing only entities/topics within the time window.

        This prevents Agent4 from seeing KG content unrelated to the
        current agenda item when multiple agendas share the same topic_id.
        """
        import copy

        lo = max(start_sec - margin_sec, 0)
        hi = end_sec + margin_sec

        def _ts_in_range(ts_raw: Any) -> bool:
            """Check whether a timestamp value falls inside [lo, hi]."""
            ts_str = str(ts_raw or "")
            if ":" in ts_str:
                ts_val = hms_to_sec(ts_str)
            else:
                try:
                    ts_val = int(float(ts_str))
                except (ValueError, TypeError):
                    return True  # keep items without parseable timestamp
            return lo <= ts_val <= hi

        out = copy.deepcopy(kg)

        # --- filter entities ---
        entities = out.get("entities", {})
        if isinstance(entities, dict):
            for key in ("financials", "decisions", "action_items", "issues"):
                items = entities.get(key, [])
                if isinstance(items, list):
                    entities[key] = [it for it in items if _ts_in_range(it.get("timestamp", ""))]
            
            # Remove high-volume global entities without timestamps
            # to prevent context dilution/bleeding across distinct agendas.
            for key in ("people", "projects", "equipment"):
                if key in entities:
                    del entities[key]

        # --- filter topics ---
        topics = out.get("topics", [])
        if isinstance(topics, list):
            filtered_topics = []
            for t in topics:
                if not isinstance(t, dict):
                    continue
                t_start = hms_to_sec(str(t.get("start_timestamp", "00:00:00")))
                t_end = hms_to_sec(str(t.get("end_timestamp", "00:00:00")))
                # keep topic if its time range overlaps with [lo, hi]
                if t_end >= lo and t_start <= hi:
                    filtered_topics.append(t)
            out["topics"] = filtered_topics

        return out

    def _agent4_topic_fallback(self, job: dict[str, Any]) -> dict[str, Any]:
        idx = int(job.get("idx", 0) or 0)
        topic_item = job.get("topic_item", {}) if isinstance(job.get("topic_item"), dict) else {}
        tl_snip = job.get("tl_snip", []) if isinstance(job.get("tl_snip"), list) else []
        slides_snip = job.get("slides_snip", []) if isinstance(job.get("slides_snip"), list) else []
        start_hms = str(job.get("start_hms", "00:00:00") or "00:00:00")
        end_hms = str(job.get("end_hms", "00:00:00") or "00:00:00")

        key_points: list[str] = []
        for row in tl_snip:
            if not isinstance(row, dict):
                continue
            text = str(row.get("text", "") or "").strip()
            if not text:
                continue
            trimmed = text[:180]
            if trimmed in key_points:
                continue
            key_points.append(trimmed)
            if len(key_points) >= 4:
                break
        if not key_points:
            key_points = ["ไม่พบข้อความสำคัญในช่วงเวลานี้ (fallback)"]

        summary_th = " | ".join(key_points[:2])
        return {
            "topic_id": topic_item.get("topic_id", f"T{idx:03d}"),
            "agenda_number": topic_item.get("agenda_number", str(idx)),
            "title": topic_item.get("title", ""),
            "department": topic_item.get("department", ""),
            "presenter": topic_item.get("key_speaker", ""),
            "time_range": f"{start_hms} – {end_hms}",
            "status": topic_item.get("status", "discussed"),
            "summary_th": summary_th,
            "key_data_points": key_points,
            "decisions": [],
            "action_items": [],
            "slide_count": len(slides_snip),
        }

    def node_agent4(self, state: WorkflowState) -> WorkflowState:
        run_meta = dict(state["run_meta"])
        artifact_dir = state.get("artifact_dir")
        topic_map = state["topic_map"]
        kg = state["kg"]
        cleaned = state["cleaned"]
        timeline = cleaned.get("timeline", [])
        slides = cleaned.get("slides", []) if isinstance(cleaned.get("slides"), list) else []

        topic_items: list[dict[str, Any]] = []
        if "agenda_mapping" in topic_map:
            for item in topic_map.get("agenda_mapping", []):
                if not isinstance(item, dict):
                    continue
                mapped = item.get("mapped_topics", []) if isinstance(item.get("mapped_topics"), list) else []
                topic_items.append(
                    {
                        "topic_id": str(mapped[0]) if mapped else "",
                        "agenda_number": str(item.get("agenda_number", "") or ""),
                        "title": str(item.get("agenda_title", "") or ""),
                        "department": str(item.get("agenda_department", "") or ""),
                        "status": str(item.get("status", "") or ""),
                        "time_range": item.get("time_range", {}),
                        "key_speaker": str(item.get("key_speaker", "") or ""),
                    }
                )
        else:
            for item in topic_map.get("extracted_topics", []):
                if not isinstance(item, dict):
                    continue
                topic_items.append(
                    {
                        "topic_id": str(item.get("id", "") or ""),
                        "agenda_number": str(item.get("number", "") or ""),
                        "title": str(item.get("title", "") or ""),
                        "department": str(item.get("department", "") or ""),
                        "status": "discussed",
                        "time_range": {
                            "start": str(item.get("start_timestamp", "00:00:00")),
                            "end": str(item.get("end_timestamp", "00:00:00")),
                        },
                        "key_speaker": ", ".join(item.get("key_speakers", [])) if isinstance(item.get("key_speakers"), list) else "",
                    }
                )
        topic_items.sort(key=lambda x: self._agenda_sort_key(x.get("agenda_number", "")))
        if "agenda_mapping" in topic_map:
            agenda_numbers = {str(x.get("agenda_number", "") or "").strip() for x in topic_items}
            container_numbers = [
                str(x.get("agenda_number", "") or "").strip()
                for x in topic_items
                if self._is_container_agenda_item(x, agenda_numbers)
            ]
            if container_numbers:
                skip_set = set(container_numbers)
                topic_items = [x for x in topic_items if str(x.get("agenda_number", "") or "").strip() not in skip_set]
                self._append_log(
                    run_meta,
                    artifact_dir,
                    "Agent4 container agendas skipped",
                    count=len(skip_set),
                    agendas=",".join(sorted(skip_set, key=self._agenda_sort_key)),
                )
        workers = self._effective_workers(len(topic_items))
        self._append_log(
            run_meta,
            artifact_dir,
            "Agent4 topic summary start",
            topics=len(topic_items),
            workers=workers,
        )

        topic_jobs: list[dict[str, Any]] = []
        for idx, topic_item in enumerate(topic_items, start=1):
            tr = topic_item.get("time_range", {})
            start_hms = str(tr.get("start", "00:00:00") if isinstance(tr, dict) else "00:00:00")
            end_hms = str(tr.get("end", "00:00:00") if isinstance(tr, dict) else "00:00:00")
            start_sec = hms_to_sec(start_hms)
            end_sec = hms_to_sec(end_hms)
            if end_sec < start_sec:
                end_sec = start_sec

            tl_snip = timeline_snippet_by_range(timeline, start_sec, end_sec)
            slides_snip = [
                s
                for s in slides
                if start_sec - 120 <= hms_to_sec(str(s.get("timestamp_hms", "00:00:00"))) <= end_sec + 120
            ]
            self._append_log(
                run_meta,
                artifact_dir,
                "Agent4 topic chunk",
                chunk=f"{idx}/{len(topic_items)}",
                topic_id=topic_item.get("topic_id", ""),
                timeline=len(tl_snip),
                slides=len(slides_snip),
            )
            # Filter KG to only include entities/topics relevant to this time range.
            filtered_kg = self._filter_kg_for_time_range(kg, start_sec, end_sec)
            topic_jobs.append(
                {
                    "idx": idx,
                    "topic_item": topic_item,
                    "start_hms": start_hms,
                    "end_hms": end_hms,
                    "slides_snip": slides_snip,
                    "tl_snip": tl_snip,
                    "filtered_kg": filtered_kg,
                }
            )

        def run_one_topic(job: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            idx = int(job["idx"])
            topic_item = job["topic_item"]
            user = fill_template(
                AGENT4_TOPIC_USR,
                KG=json.dumps(sanitize_kg_for_output(job["filtered_kg"]), ensure_ascii=False),
                TOPIC_ITEM=json.dumps(topic_item, ensure_ascii=False),
                TIMELINE_SNIPPET=json.dumps(job["tl_snip"], ensure_ascii=False),
                SLIDES=json.dumps(job["slides_snip"], ensure_ascii=False),
            )
            out = self.llm.call(
                AGENT4_TOPIC_SYS,
                user,
                json_mode=True,
                required_keys=["topic_summary"],
                tag=f"agent4_topic_{idx}",
            )
            assert isinstance(out, dict)
            ts = out.get("topic_summary", {})
            if not isinstance(ts, dict):
                ts = {}
            # Keep structural fields aligned with agenda/topic mapping, regardless of LLM drift.
            ts["topic_id"] = topic_item.get("topic_id", f"T{idx:03d}")
            ts["agenda_number"] = topic_item.get("agenda_number", str(idx))
            ts["title"] = topic_item.get("title", "")
            ts["department"] = topic_item.get("department", "")
            ts["presenter"] = topic_item.get("key_speaker", "")
            ts["time_range"] = f"{job['start_hms']} – {job['end_hms']}"
            ts["status"] = topic_item.get("status", "discussed")

            summary_th = ts.get("summary_th", "")
            ts["summary_th"] = summary_th if isinstance(summary_th, str) else str(summary_th or "")

            for key in ["key_data_points", "decisions", "action_items"]:
                if not isinstance(ts.get(key), list):
                    ts[key] = []

            ts["slide_count"] = len(job["slides_snip"])
            return idx, ts

        topic_summaries: list[dict[str, Any]] = []
        if workers == 1:
            for job in topic_jobs:
                idx = int(job["idx"])
                try:
                    _, ts = run_one_topic(job)
                except Exception as exc:
                    self._append_log(
                        run_meta,
                        artifact_dir,
                        "Agent4 topic failed",
                        chunk=f"{idx}/{len(topic_items)}",
                        error=str(exc),
                        action="deterministic_fallback",
                    )
                    ts = self._agent4_topic_fallback(job)
                topic_summaries.append(ts)
                self._append_log(
                    run_meta,
                    artifact_dir,
                    "Agent4 topic done",
                    chunk=f"{idx}/{len(topic_items)}",
                    decisions=len(ts.get("decisions", [])),
                    actions=len(ts.get("action_items", [])),
                )
        else:
            results: dict[int, dict[str, Any]] = {}
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(run_one_topic, job): job for job in topic_jobs}
                for future in as_completed(futures):
                    job = futures[future]
                    idx = int(job["idx"])
                    try:
                        _, ts = future.result()
                    except Exception as exc:
                        self._append_log(
                            run_meta,
                            artifact_dir,
                            "Agent4 topic failed",
                            chunk=f"{idx}/{len(topic_items)}",
                            error=str(exc),
                            action="deterministic_fallback",
                        )
                        ts = self._agent4_topic_fallback(job)
                    results[idx] = ts
                    self._append_log(
                        run_meta,
                        artifact_dir,
                        "Agent4 topic done",
                        chunk=f"{idx}/{len(topic_items)}",
                        decisions=len(ts.get("decisions", [])),
                        actions=len(ts.get("action_items", [])),
                    )
            for idx in sorted(results):
                topic_summaries.append(results[idx])

        max_transcript_sec = max(float(s.get("end", 0) or 0) for s in state["segments"])
        self._append_log(run_meta, artifact_dir, "Agent4 executive summary start")
        exec_user = fill_template(
            AGENT4_EXEC_USR,
            TOPIC_SUMMARIES=json.dumps(topic_summaries, ensure_ascii=False),
            KG=json.dumps(sanitize_kg_for_output(kg), ensure_ascii=False),
        )
        try:
            exec_out = self.llm.call(
                AGENT4_EXEC_SYS,
                exec_user,
                json_mode=True,
                required_keys=["executive_summary_th", "total_decisions", "total_action_items", "meeting_duration"],
                tag="agent4_exec",
            )
            assert isinstance(exec_out, dict)
        except Exception as exc:
            self._append_log(
                run_meta,
                artifact_dir,
                "Agent4 executive summary failed",
                error=str(exc),
                action="deterministic_fallback",
            )
            fallback_bullets: list[str] = []
            for item in topic_summaries:
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title", "") or "").strip()
                summary = str(item.get("summary_th", "") or "").strip()
                if not (title or summary):
                    continue
                line = f"{title}: {summary}" if title else summary
                fallback_bullets.append(line[:180])
                if len(fallback_bullets) >= 5:
                    break
            fallback_text = "\n".join(f"- {x}" for x in fallback_bullets)
            exec_out = {
                "executive_summary_th": (
                    "สรุปผู้บริหาร (fallback): ระบบใช้ deterministic fallback เพราะ LLM ตอบไม่เข้า schema.\n"
                    + fallback_text
                ).strip(),
                "total_decisions": len(kg.get("entities", {}).get("decisions", [])),
                "total_action_items": len(kg.get("entities", {}).get("action_items", [])),
                "meeting_duration": sec_to_hms(max_transcript_sec),
            }

        total_decisions = int(exec_out.get("total_decisions", 0) or 0)
        total_actions = int(exec_out.get("total_action_items", 0) or 0)

        if total_decisions <= 0:
            total_decisions = len(kg.get("entities", {}).get("decisions", []))
        if total_actions <= 0:
            total_actions = len(kg.get("entities", {}).get("action_items", []))

        summaries = {
            "topic_summaries": topic_summaries,
            "executive_summary_th": str(exec_out.get("executive_summary_th", "")),
            "total_decisions": total_decisions,
            "total_action_items": total_actions,
            "meeting_duration": str(exec_out.get("meeting_duration", sec_to_hms(max_transcript_sec))),
        }
        self._append_log(
            run_meta,
            artifact_dir,
            "Agent4 done",
            topic_summaries=len(topic_summaries),
            total_decisions=total_decisions,
            total_action_items=total_actions,
        )

        self._save_json_if_enabled(state, "agent4_summaries.json", summaries)
        return {"summaries": summaries, "run_meta": run_meta}

    def node_agent5(self, state: WorkflowState) -> WorkflowState:
        run_meta = dict(state["run_meta"])
        artifact_dir = state.get("artifact_dir")
        self._append_log(run_meta, artifact_dir, "Agent5 start", layout=self.cfg.report_layout_mode)
        cleaned = state["cleaned"]
        summaries = state["summaries"]
        kg = state["kg"]
        image_by_topic = state.get("image_by_topic", {})
        topic_rows = summaries.get("topic_summaries", []) if isinstance(summaries.get("topic_summaries"), list) else []
        summary_topic_ids = {
            str(t.get("topic_id", "") or "")
            for t in topic_rows
            if isinstance(t, dict) and str(t.get("topic_id", "") or "")
        }
        image_topic_ids = {
            str(topic_id)
            for topic_id, rows in image_by_topic.items()
            if isinstance(rows, list) and rows
        }
        image_items = sum(len(rows) for rows in image_by_topic.values() if isinstance(rows, list))
        unmatched_image_items = sum(
            len(rows)
            for topic_id, rows in image_by_topic.items()
            if isinstance(rows, list) and str(topic_id) not in summary_topic_ids
        )
        self._append_log(
            run_meta,
            artifact_dir,
            "Agent5 image coverage",
            summary_topics=len(summary_topic_ids),
            image_topics=len(image_topic_ids),
            overlap_topics=len(summary_topic_ids & image_topic_ids),
            image_items=image_items,
            unmatched_image_items=unmatched_image_items,
        )

        safe_kg = sanitize_kg_for_output(kg)
        agent5_user = fill_template(
            AGENT5_USR,
            META=json.dumps(cleaned.get("meeting_meta", {}), ensure_ascii=False),
            SUMMARIES=json.dumps(summaries, ensure_ascii=False),
            KG=json.dumps(safe_kg, ensure_ascii=False),
            IMAGE_BY_TOPIC=json.dumps(image_by_topic, ensure_ascii=False),
            FULL_CSS_JS=HTML_CSS_JS_BUNDLE,
        )

        def render_fallback_html() -> tuple[str, str]:
            return (
                fallback_render_html(
                    cleaned.get("meeting_meta", {}),
                    summaries,
                    safe_kg,
                    image_by_topic,
                ),
                "fallback_current",
            )

        html_source = "llm"
        if self.llm.typhoon_llm is None:
            self._append_log(
                run_meta,
                artifact_dir,
                "Agent5 llm skipped",
                reason="ollama_only_mode",
                action="fallback_renderer",
            )
            html, html_source = render_fallback_html()
        else:
            try:
                html = self.llm.call(
                    AGENT5_SYS,
                    agent5_user,
                    json_mode=False,
                    required_keys=None,
                    tag="agent5_html",
                )
                assert isinstance(html, str)
                self._append_log(run_meta, artifact_dir, "Agent5 llm done", html_chars=len(html))
                self._save_html_if_enabled(state, "agent5_raw_llm.html", html)

                normalized_html = strip_markdown_fences(html)
                if normalized_html != html:
                    self._append_log(run_meta, artifact_dir, "Agent5 html normalized", stripped_markdown_fence=True)
                html = normalized_html

                compliance_issues = html_compliance_issues(
                    html,
                    expected_topic_sections=len(topic_rows),
                )
            except Exception as exc:
                self._append_log(
                    run_meta,
                    artifact_dir,
                    "Agent5 llm failed",
                    action="fallback_renderer",
                    error=str(exc),
                )
                html, html_source = render_fallback_html()
                compliance_issues = []

            if compliance_issues:
                self._append_log(
                    run_meta,
                    artifact_dir,
                    "Agent5 compliance failed",
                    action="fallback_renderer",
                    issues="; ".join(compliance_issues[:3]),
                )
                html, html_source = render_fallback_html()

        if self.cfg.report_layout_mode == "react_official":
            html = apply_react_official_theme(html)
            self._append_log(
                run_meta,
                artifact_dir,
                "Agent5 theme applied",
                theme="react_official",
                mode="override",
                html_chars=len(html),
            )

        Path(self.cfg.output_html_path).write_text(html, encoding="utf-8")
        self._save_html_if_enabled(state, "agent5_report.html", html)

        run_meta["provider_calls"] = self.llm.call_log
        run_meta["finished_at"] = datetime.now().isoformat()
        run_meta["output_html_path"] = str(Path(self.cfg.output_html_path).resolve())

        self._append_log(run_meta, artifact_dir, "Done", output=self.cfg.output_html_path)
        self._append_log(
            run_meta,
            artifact_dir,
            f"output path : {Path(self.cfg.output_html_path).resolve()}",
        )
        if self.cfg.save_intermediate:
            self._append_log(run_meta, artifact_dir, "Artifacts saved", path=state["artifact_dir"])
            save_json(self._artifact_path(state, "run_metadata.json"), run_meta)

        return {"html": html, "run_meta": run_meta}

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(WorkflowState)
        graph.add_node("load_inputs", self.node_load_inputs)
        graph.add_node("agent1", self.node_agent1)
        graph.add_node("agent2", self.node_agent2)
        graph.add_node("agent3a", self.node_agent3a)
        graph.add_node("agent3b", self.node_agent3b)
        graph.add_node("agent25", self.node_agent25)
        graph.add_node("agent4", self.node_agent4)
        graph.add_node("agent5", self.node_agent5)

        graph.set_entry_point("load_inputs")
        graph.add_edge("load_inputs", "agent1")
        graph.add_edge("agent1", "agent2")
        graph.add_conditional_edges(
            "agent2",
            self.route_after_agent2,
            {"agent3a": "agent3a", "agent3b": "agent3b"},
        )
        graph.add_edge("agent3a", "agent25")
        graph.add_edge("agent3b", "agent25")
        graph.add_edge("agent25", "agent4")
        graph.add_edge("agent4", "agent5")
        graph.add_edge("agent5", END)
        return graph

    def run(self) -> WorkflowState:
        return self.graph.invoke({})
