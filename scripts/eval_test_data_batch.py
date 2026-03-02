#!/usr/bin/env python3
"""Batch runner/evaluator for datasets under test_data/.

For each dataset id (filename stem shared by transcript/ and meetconfig/):
1) Optionally run orchestrator.py to generate HTML.
2) Measure time coverage from transcript vs rendered HTML timestamps.
3) Optionally ask Gemini if summary content aligns with agenda topics.
4) Write one JSON report for all datasets.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import URLError, HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


HMS_RE = re.compile(r"\b(\d{2}):(\d{2}):(\d{2})\b")
ARTIFACT_PATH_RE = re.compile(r"artifact path\s*:\s*(.+)$", flags=re.IGNORECASE | re.MULTILINE)


@dataclass
class TimeRange:
    min_sec: float | None
    max_sec: float | None

    @property
    def duration_sec(self) -> float | None:
        if self.min_sec is None or self.max_sec is None:
            return None
        return max(0.0, self.max_sec - self.min_sec)

    def to_json(self) -> dict[str, Any]:
        return {
            "min_sec": self.min_sec,
            "max_sec": self.max_sec,
            "min_hms": sec_to_hms(self.min_sec) if self.min_sec is not None else None,
            "max_hms": sec_to_hms(self.max_sec) if self.max_sec is not None else None,
            "duration_sec": self.duration_sec,
        }


def sec_to_hms(value: float | int | None) -> str:
    if value is None:
        return ""
    total = max(int(float(value)), 0)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def hms_to_sec(hms: str) -> int:
    parts = [int(p) for p in str(hms).split(":")]
    if len(parts) != 3:
        return 0
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def strip_html_tags(html: str) -> str:
    # Keep this lightweight with regex; good enough for LLM prompt context.
    text = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", html)
    text = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_time_range_from_transcript(transcript_path: Path) -> TimeRange:
    obj = json.loads(transcript_path.read_text(encoding="utf-8"))
    segments = obj.get("segments", [])
    if not isinstance(segments, list):
        return TimeRange(None, None)
    starts: list[float] = []
    ends: list[float] = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        try:
            starts.append(float(seg.get("start", 0) or 0))
        except Exception:
            pass
        try:
            ends.append(float(seg.get("end", seg.get("start", 0)) or 0))
        except Exception:
            pass
    if not starts or not ends:
        return TimeRange(None, None)
    return TimeRange(min(starts), max(ends))


def extract_time_range_from_text(text: str) -> tuple[TimeRange, int, int]:
    times = [f"{h}:{m}:{s}" for h, m, s in HMS_RE.findall(text)]
    if not times:
        return TimeRange(None, None), 0, 0
    sec_values = [hms_to_sec(t) for t in times]
    return TimeRange(min(sec_values), max(sec_values)), len(times), len(set(times))


def compute_coverage_percent(source: TimeRange, observed: TimeRange) -> float | None:
    if source.min_sec is None or source.max_sec is None:
        return None
    if observed.min_sec is None or observed.max_sec is None:
        return 0.0
    source_dur = max(0.0, source.max_sec - source.min_sec)
    if source_dur <= 0:
        return None
    overlap_start = max(source.min_sec, observed.min_sec)
    overlap_end = min(source.max_sec, observed.max_sec)
    overlap = max(0.0, overlap_end - overlap_start)
    return round((overlap / source_dur) * 100.0, 2)


def discover_dataset_ids(transcript_dir: Path, config_dir: Path) -> list[str]:
    transcript_ids = {p.stem for p in transcript_dir.glob("*.json")}
    config_ids = {p.stem for p in config_dir.glob("*.json")}
    return sorted(transcript_ids & config_ids)


def parse_artifact_path(log_text: str) -> str | None:
    m = ARTIFACT_PATH_RE.search(log_text or "")
    if not m:
        return None
    return m.group(1).strip()


def find_topic_map_range(artifact_dir: Path | None) -> TimeRange:
    if not artifact_dir:
        return TimeRange(None, None)
    topic_map_path = artifact_dir / "agent3_topic_map.json"
    if not topic_map_path.exists():
        return TimeRange(None, None)

    obj = json.loads(topic_map_path.read_text(encoding="utf-8"))
    items = []
    if isinstance(obj.get("agenda_mapping"), list):
        items = obj.get("agenda_mapping", [])
    elif isinstance(obj.get("extracted_topics"), list):
        items = obj.get("extracted_topics", [])

    starts: list[int] = []
    ends: list[int] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        tr = it.get("time_range")
        if isinstance(tr, dict):
            s = str(tr.get("start", "") or "")
            e = str(tr.get("end", "") or "")
        else:
            s = str(it.get("start_timestamp", "") or "")
            e = str(it.get("end_timestamp", "") or "")
        if not s or not e:
            continue
        starts.append(hms_to_sec(s))
        ends.append(hms_to_sec(e))
    if not starts or not ends:
        return TimeRange(None, None)
    return TimeRange(float(min(starts)), float(max(ends)))


def call_gemini_alignment(
    *,
    api_key: str,
    model: str,
    agenda_text: str,
    html_text: str,
    dataset_id: str,
    timeout_sec: int,
) -> dict[str, Any]:
    prompt = (
        "You are evaluating meeting summary quality.\n"
        "Task: Check whether summary content matches agenda topics.\n"
        "Return strict JSON only with keys:\n"
        "{"
        '"alignment_score_0_100": number, '
        '"is_aligned": boolean, '
        '"issues": [string], '
        '"matched_topics": [string], '
        '"missing_topics": [string], '
        '"notes": string'
        "}\n\n"
        f"DATASET_ID:\n{dataset_id}\n\n"
        "AGENDA_TEXT:\n"
        f"{agenda_text[:18000]}\n\n"
        "SUMMARY_HTML_TEXT:\n"
        f"{html_text[:26000]}"
    )

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?"
        + urlencode({"key": api_key})
    )
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.1,
        },
    }
    req = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(req, timeout=timeout_sec) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
        return {"error": f"gemini_http_error: {exc.code}", "detail": text[:2000]}
    except URLError as exc:
        return {"error": "gemini_url_error", "detail": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"error": "gemini_call_error", "detail": str(exc)}

    try:
        obj = json.loads(body)
    except Exception:
        return {"error": "gemini_non_json_response", "raw": body[:2000]}

    candidates = obj.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        return {"error": "gemini_no_candidates", "raw": obj}

    parts = (
        candidates[0].get("content", {}).get("parts", [])
        if isinstance(candidates[0], dict)
        else []
    )
    text_out = ""
    if isinstance(parts, list):
        for part in parts:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                text_out += part["text"]
    text_out = text_out.strip()

    # Gemini sometimes wraps JSON in markdown code fences.
    cleaned = re.sub(r"^\s*```(?:json)?\s*", "", text_out, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\s*```\s*$", "", cleaned).strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
        return {"error": "gemini_result_not_object", "raw": cleaned[:2000]}
    except Exception:
        return {"error": "gemini_result_parse_error", "raw": cleaned[:2000]}


def run_one_dataset(
    *,
    dataset_id: str,
    transcript_path: Path,
    config_path: Path,
    html_path: Path,
    output_root: Path,
    run_orchestrator: bool,
    mode: str,
    report_layout: str,
    save_artifacts: bool,
    include_ocr: bool,
    ocr_path: Path | None,
    per_run_timeout_sec: int,
    gemini_api_key: str | None,
    gemini_model: str,
    gemini_timeout_sec: int,
    skip_gemini: bool,
) -> dict[str, Any]:
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    run_log_path = log_dir / f"{dataset_id}.log"
    artifact_dir: Path | None = None

    started = time.time()
    proc_exit_code: int | None = None
    run_status = "not_run"
    run_error: str | None = None

    if run_orchestrator:
        env = os.environ.copy()
        env["TRANSCRIPT_PATH"] = str(transcript_path)
        env["CONFIG_PATH"] = str(config_path)
        env["OUTPUT_HTML_PATH"] = str(html_path)
        env["INCLUDE_OCR"] = "true" if include_ocr else "false"
        if include_ocr and ocr_path:
            env["OCR_PATH"] = str(ocr_path)

        cmd = [
            sys.executable,
            "orchestrator.py",
            "--mode",
            mode,
            "--report-layout",
            report_layout,
            "--output",
            str(html_path),
            "--save-artifacts",
            "true" if save_artifacts else "false",
        ]

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(Path(__file__).resolve().parents[1]),
                env=env,
                capture_output=True,
                text=True,
                timeout=per_run_timeout_sec,
            )
            combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
            run_log_path.write_text(combined, encoding="utf-8")
            proc_exit_code = proc.returncode
            parsed_artifact = parse_artifact_path(combined)
            if parsed_artifact:
                artifact_dir = Path(parsed_artifact)
            run_status = "succeeded" if proc.returncode == 0 and html_path.exists() else "failed"
            if run_status == "failed":
                run_error = f"orchestrator_exit_code={proc.returncode}"
        except subprocess.TimeoutExpired:
            run_status = "failed"
            proc_exit_code = 124
            run_error = f"orchestrator_timeout>{per_run_timeout_sec}s"
            partial_stdout = ""
            partial_stderr = ""
            try:
                partial_stdout = str(getattr(exc, "stdout", "") or "")
            except Exception:
                partial_stdout = ""
            try:
                partial_stderr = str(getattr(exc, "stderr", "") or "")
            except Exception:
                partial_stderr = ""
            partial_combined = (partial_stdout + "\n" + partial_stderr).strip()
            if partial_combined:
                parsed_artifact = parse_artifact_path(partial_combined)
                if parsed_artifact:
                    artifact_dir = Path(parsed_artifact)
                timeout_log = (
                    f"{run_error}\n"
                    "--- partial stdout/stderr before timeout ---\n"
                    f"{partial_combined}\n"
                )
                run_log_path.write_text(timeout_log, encoding="utf-8")
            else:
                run_log_path.write_text(run_error, encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            run_status = "failed"
            proc_exit_code = -1
            run_error = f"orchestrator_exception: {exc}"
            run_log_path.write_text(run_error, encoding="utf-8")

    runtime_sec = round(time.time() - started, 3)

    transcript_range = extract_time_range_from_transcript(transcript_path)

    html_range = TimeRange(None, None)
    html_time_tokens = 0
    html_unique_times = 0
    html_text = ""
    if html_path.exists():
        raw_html = html_path.read_text(encoding="utf-8", errors="ignore")
        html_text = strip_html_tags(raw_html)
        html_range, html_time_tokens, html_unique_times = extract_time_range_from_text(raw_html)

    topic_map_range = find_topic_map_range(artifact_dir)

    html_coverage_pct = compute_coverage_percent(transcript_range, html_range)
    topic_map_coverage_pct = compute_coverage_percent(transcript_range, topic_map_range)

    gemini_result: dict[str, Any] | None = None
    if not skip_gemini:
        if not gemini_api_key:
            gemini_result = {"error": "missing_gemini_api_key"}
        elif not html_text:
            gemini_result = {"error": "missing_html_output"}
        else:
            cfg_obj = json.loads(config_path.read_text(encoding="utf-8"))
            agenda_text = str(cfg_obj.get("AGENDA_TEXT", "") or "")
            gemini_result = call_gemini_alignment(
                api_key=gemini_api_key,
                model=gemini_model,
                agenda_text=agenda_text,
                html_text=html_text,
                dataset_id=dataset_id,
                timeout_sec=gemini_timeout_sec,
            )

    return {
        "dataset_id": dataset_id,
        "paths": {
            "transcript": str(transcript_path),
            "config": str(config_path),
            "html": str(html_path),
            "artifact_dir": str(artifact_dir) if artifact_dir else None,
            "run_log": str(run_log_path),
        },
        "run": {
            "status": run_status,
            "exit_code": proc_exit_code,
            "runtime_sec": runtime_sec,
            "error": run_error,
        },
        "time_ranges": {
            "transcript": transcript_range.to_json(),
            "html": html_range.to_json(),
            "topic_map": topic_map_range.to_json(),
        },
        "coverage": {
            "html_coverage_pct": html_coverage_pct,
            "topic_map_coverage_pct": topic_map_coverage_pct,
            "html_time_tokens": html_time_tokens,
            "html_unique_times": html_unique_times,
        },
        "gemini_alignment": gemini_result,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-evaluate test_data datasets.")
    parser.add_argument(
        "--test-data-dir",
        default="./test_data",
        help="Root directory containing transcript/ and meetconfig/.",
    )
    parser.add_argument(
        "--output-root",
        default="./output/test_data_eval",
        help="Output root for html/log/summary files.",
    )
    parser.add_argument(
        "--summary-json",
        default="",
        help="Summary JSON output path (default: <output-root>/summary.json).",
    )
    parser.add_argument("--mode", choices=["agenda", "auto"], default="agenda")
    parser.add_argument("--report-layout", choices=["current", "react_official"], default="react_official")
    parser.add_argument("--save-artifacts", choices=["true", "false"], default="true")
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run orchestrator for each dataset before evaluation.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of datasets (0 = all).",
    )
    parser.add_argument(
        "--ocr-path",
        default="",
        help="Optional OCR file to use for all datasets. If empty, OCR is disabled.",
    )
    parser.add_argument("--per-run-timeout-sec", type=int, default=7200)
    parser.add_argument("--skip-gemini", action="store_true")
    parser.add_argument("--gemini-api-key", default=os.getenv("GEMINI_API_KEY", ""))
    parser.add_argument("--gemini-model", default=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"))
    parser.add_argument("--gemini-timeout-sec", type=int, default=90)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    test_data_dir = (repo_root / args.test_data_dir).resolve()
    transcript_dir = test_data_dir / "transcript"
    config_dir = test_data_dir / "meetconfig"
    output_root = (repo_root / args.output_root).resolve()
    html_dir = output_root / "html"
    html_dir.mkdir(parents=True, exist_ok=True)

    summary_json_path = (
        Path(args.summary_json).resolve()
        if args.summary_json.strip()
        else output_root / "summary.json"
    )
    summary_json_path.parent.mkdir(parents=True, exist_ok=True)

    if not transcript_dir.exists() or not config_dir.exists():
        print(f"Missing test_data dirs: {transcript_dir} or {config_dir}", file=sys.stderr)
        return 2

    dataset_ids = discover_dataset_ids(transcript_dir, config_dir)
    if args.limit > 0:
        dataset_ids = dataset_ids[: args.limit]
    if not dataset_ids:
        print("No matching datasets found.", file=sys.stderr)
        return 2

    ocr_path = Path(args.ocr_path).resolve() if args.ocr_path.strip() else None
    include_ocr = bool(ocr_path and ocr_path.exists())
    save_artifacts = args.save_artifacts.lower() == "true"

    results: list[dict[str, Any]] = []
    for idx, dataset_id in enumerate(dataset_ids, start=1):
        transcript_path = transcript_dir / f"{dataset_id}.json"
        config_path = config_dir / f"{dataset_id}.json"
        html_path = html_dir / f"{dataset_id}.html"
        print(f"[{idx}/{len(dataset_ids)}] dataset={dataset_id} run={args.run}")
        row = run_one_dataset(
            dataset_id=dataset_id,
            transcript_path=transcript_path,
            config_path=config_path,
            html_path=html_path,
            output_root=output_root,
            run_orchestrator=args.run,
            mode=args.mode,
            report_layout=args.report_layout,
            save_artifacts=save_artifacts,
            include_ocr=include_ocr,
            ocr_path=ocr_path,
            per_run_timeout_sec=args.per_run_timeout_sec,
            gemini_api_key=args.gemini_api_key.strip() or None,
            gemini_model=args.gemini_model.strip(),
            gemini_timeout_sec=args.gemini_timeout_sec,
            skip_gemini=args.skip_gemini,
        )
        results.append(row)

    succeeded = sum(1 for r in results if r.get("run", {}).get("status") == "succeeded")
    failed = sum(1 for r in results if r.get("run", {}).get("status") == "failed")

    coverage_values = [
        float(r.get("coverage", {}).get("html_coverage_pct"))
        for r in results
        if isinstance(r.get("coverage", {}).get("html_coverage_pct"), (int, float))
    ]
    avg_coverage = round(sum(coverage_values) / len(coverage_values), 2) if coverage_values else None

    summary: dict[str, Any] = {
        "generated_at_epoch": int(time.time()),
        "config": {
            "test_data_dir": str(test_data_dir),
            "output_root": str(output_root),
            "run": args.run,
            "mode": args.mode,
            "report_layout": args.report_layout,
            "include_ocr": include_ocr,
            "ocr_path": str(ocr_path) if ocr_path else None,
            "gemini_model": None if args.skip_gemini else args.gemini_model,
        },
        "stats": {
            "dataset_count": len(results),
            "run_succeeded": succeeded,
            "run_failed": failed,
            "avg_html_coverage_pct": avg_coverage,
        },
        "results": results,
    }

    summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote summary: {summary_json_path}")
    print(f"Datasets={len(results)} succeeded={succeeded} failed={failed} avg_cov={avg_coverage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
