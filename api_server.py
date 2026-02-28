from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import sys
import threading
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, ValidationError, model_validator

from pipeline_utils import hms_to_sec, sec_to_hms


def _env_int(name: str, default: int, min_value: int = 0) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return max(int(raw), min_value)
    except Exception:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_csv(name: str, default: str) -> list[str]:
    raw = os.getenv(name, default)
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or ["*"]


def _configure_logger() -> logging.Logger:
    level_name = str(os.getenv("API_LOG_LEVEL", "INFO")).strip().upper()
    level = getattr(logging, level_name, logging.INFO)
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.WARNING,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        )
    logger = logging.getLogger("meeting_api")
    logger.setLevel(level)
    return logger


LOGGER = _configure_logger()

API_MAX_REQUEST_BODY_BYTES = _env_int("API_MAX_REQUEST_BODY_BYTES", 5 * 1024 * 1024, min_value=1024)
API_MAX_SEGMENTS = _env_int("API_MAX_SEGMENTS", 10_000, min_value=1)
API_MAX_FULL_TEXT_CHARS = _env_int("API_MAX_FULL_TEXT_CHARS", 1_500_000, min_value=1)
API_MAX_MEETING_INFO_CHARS = _env_int("API_MAX_MEETING_INFO_CHARS", 200_000, min_value=1)
API_MAX_AGENDA_TEXT_CHARS = _env_int("API_MAX_AGENDA_TEXT_CHARS", 500_000, min_value=1)
API_MAX_TOPIC_TIME_OVERRIDES = _env_int("API_MAX_TOPIC_TIME_OVERRIDES", 2_000, min_value=1)
API_MAX_CAPTURES = _env_int("API_MAX_CAPTURES", 30_000, min_value=1)
API_WORKER_JOIN_TIMEOUT_SEC = _env_int("API_WORKER_JOIN_TIMEOUT_SEC", 10, min_value=1)
API_PROCESS_TERMINATE_GRACE_SEC = _env_int("API_PROCESS_TERMINATE_GRACE_SEC", 5, min_value=1)
API_JOBS_ROOT = str(os.getenv("API_JOBS_ROOT", "")).strip()
API_CORS_ALLOW_ORIGINS = _env_csv("API_CORS_ALLOW_ORIGINS", "*")
API_CORS_ALLOW_METHODS = _env_csv("API_CORS_ALLOW_METHODS", "*")
API_CORS_ALLOW_HEADERS = _env_csv("API_CORS_ALLOW_HEADERS", "*")
API_CORS_ALLOW_CREDENTIALS = _env_bool("API_CORS_ALLOW_CREDENTIALS", False)
if "*" in API_CORS_ALLOW_ORIGINS and API_CORS_ALLOW_CREDENTIALS:
    LOGGER.warning(
        "API_CORS_ALLOW_CREDENTIALS=true is incompatible with wildcard origins; forcing false."
    )
    API_CORS_ALLOW_CREDENTIALS = False


class MeetingRunRequest(BaseModel):
    MEETING_INFO: str
    segments: list[dict[str, Any]]
    full_text: str
    AGENDA_TEXT: str | None = None
    TOPIC_TIME_OVERRIDES: list[dict[str, Any]] | None = None
    capture_ocr_results: dict[str, Any] | None = None
    captures: list[dict[str, Any]] | None = None
    mode: Literal["agenda", "auto"] | None = None
    report_layout: Literal["current", "react_official"] = "react_official"
    image_insert_enabled: bool = True
    save_artifacts: bool = True
    resume_artifact_dir: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_incoming_aliases(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        out = dict(data)
        if "capture_ocr_results.json" in out and "capture_ocr_results" not in out:
            out["capture_ocr_results"] = out.pop("capture_ocr_results.json")
        return out

    @model_validator(mode="after")
    def _validate_payload_limits(self) -> "MeetingRunRequest":
        meeting_info = str(self.MEETING_INFO or "")
        if len(meeting_info) > API_MAX_MEETING_INFO_CHARS:
            raise ValueError(f"MEETING_INFO exceeds limit ({API_MAX_MEETING_INFO_CHARS} chars)")

        full_text = str(self.full_text or "")
        if len(full_text) > API_MAX_FULL_TEXT_CHARS:
            raise ValueError(f"full_text exceeds limit ({API_MAX_FULL_TEXT_CHARS} chars)")

        if len(self.segments) > API_MAX_SEGMENTS:
            raise ValueError(f"segments exceeds limit ({API_MAX_SEGMENTS} items)")

        if self.AGENDA_TEXT and len(self.AGENDA_TEXT) > API_MAX_AGENDA_TEXT_CHARS:
            raise ValueError(f"AGENDA_TEXT exceeds limit ({API_MAX_AGENDA_TEXT_CHARS} chars)")

        if self.TOPIC_TIME_OVERRIDES and len(self.TOPIC_TIME_OVERRIDES) > API_MAX_TOPIC_TIME_OVERRIDES:
            raise ValueError(
                f"TOPIC_TIME_OVERRIDES exceeds limit ({API_MAX_TOPIC_TIME_OVERRIDES} items)"
            )

        direct_captures = self.captures if isinstance(self.captures, list) else []
        embedded_captures = []
        if isinstance(self.capture_ocr_results, dict):
            raw = self.capture_ocr_results.get("captures")
            if isinstance(raw, list):
                embedded_captures = raw

        if len(direct_captures) > API_MAX_CAPTURES:
            raise ValueError(f"captures exceeds limit ({API_MAX_CAPTURES} items)")
        if len(embedded_captures) > API_MAX_CAPTURES:
            raise ValueError(f"capture_ocr_results.captures exceeds limit ({API_MAX_CAPTURES} items)")
        return self


class JobCreateResponse(BaseModel):
    job_id: str
    status: str
    queue_position: int | None = None
    status_url: str
    html_url: str
    logs_url: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    queue_position: int | None = None
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    mode: str
    report_layout: str
    error: str | None = None
    work_dir: str
    html_path: str
    log_path: str
    artifact_dir: str | None = None
    runtime_log_path: str | None = None
    status_url: str
    html_url: str
    logs_url: str


@dataclass
class JobRecord:
    job_id: str
    status: str
    created_at: str
    started_at: str | None
    finished_at: str | None
    mode: str
    report_layout: str
    image_insert_enabled: bool
    save_artifacts: bool
    resume_artifact_dir: str
    work_dir: str
    transcript_path: str
    config_path: str
    ocr_path: str
    html_path: str
    log_path: str
    artifact_dir: str | None
    runtime_log_path: str | None
    error: str | None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _tail_text(path: Path, max_lines: int = 200) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if max_lines <= 0:
        return "\n".join(lines)
    return "\n".join(lines[-max_lines:])


def _is_remote_http_path(value: str) -> bool:
    text = str(value or "").strip().lower()
    return text.startswith("http://") or text.startswith("https://")


def _pick_image_path(item: dict[str, Any]) -> str:
    choices = [
        "image_presigned_url",
        "image_url",
        "presigned_url",
        "s3_presigned_url",
        "s3_url",
        "url",
        "image_path",
    ]
    for key in choices:
        raw = str(item.get(key, "") or "").strip()
        if raw:
            return raw
    return ""


def _normalize_captures(captures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, raw in enumerate(captures, start=1):
        if not isinstance(raw, dict):
            continue
        row = dict(raw)

        capture_index = _as_int(row.get("capture_index"), idx)
        if capture_index <= 0:
            capture_index = idx
        row["capture_index"] = capture_index

        ts_sec = _as_float(row.get("timestamp_sec"), -1.0)
        ts_hms = str(row.get("timestamp_hms", "") or "").strip()
        if ts_sec < 0 and ts_hms:
            ts_sec = float(hms_to_sec(ts_hms))
        if ts_sec < 0:
            ts_sec = 0.0
        if not ts_hms:
            ts_hms = sec_to_hms(ts_sec)
        row["timestamp_sec"] = float(ts_sec)
        row["timestamp_hms"] = ts_hms

        image_path = _pick_image_path(row)
        if image_path:
            row["image_path"] = image_path

        if "ocr_text" not in row:
            row["ocr_text"] = str(row.get("text", "") or "")

        ocr_size = _as_int(row.get("ocr_file_size_bytes"), -1)
        if ocr_size < 0:
            if _is_remote_http_path(image_path):
                # Unknown content length for presigned URLs: keep images eligible.
                ocr_size = 1024 * 1024
            else:
                ocr_size = _as_int(row.get("image_size_bytes"), 0)
        row["ocr_file_size_bytes"] = ocr_size

        out.append(row)

    out.sort(key=lambda x: _as_int(x.get("capture_index"), 0))
    return out


class JobQueueManager:
    def __init__(self, project_root: Path, jobs_root: Path | None = None):
        self.project_root = project_root
        self.jobs_root = jobs_root or (project_root / "output" / "api_jobs")
        self.jobs_root.mkdir(parents=True, exist_ok=True)

        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.RLock()
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._active_job_id: str | None = None
        self._active_process: subprocess.Popen | None = None
        self._stop_event = threading.Event()
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="meeting-api-worker",
            daemon=True,
        )

    def start(self) -> None:
        if not self._worker.is_alive():
            LOGGER.info("queue worker starting")
            self._worker.start()

    def stop(self) -> None:
        LOGGER.info("queue worker stopping")
        self._stop_event.set()
        self._queue.put(None)
        if self._worker.is_alive():
            self._worker.join(timeout=API_WORKER_JOIN_TIMEOUT_SEC)

        proc = self._get_active_process()
        if self._worker.is_alive() and proc is not None and proc.poll() is None:
            LOGGER.warning(
                "worker did not stop in %ss; terminating subprocess pid=%s",
                API_WORKER_JOIN_TIMEOUT_SEC,
                proc.pid,
            )
            try:
                proc.terminate()
                proc.wait(timeout=API_PROCESS_TERMINATE_GRACE_SEC)
            except subprocess.TimeoutExpired:
                LOGGER.warning("subprocess pid=%s did not terminate; killing", proc.pid)
                proc.kill()
            except Exception:
                LOGGER.exception("failed to stop active subprocess cleanly")
            if self._worker.is_alive():
                self._worker.join(timeout=API_WORKER_JOIN_TIMEOUT_SEC)

        if self._worker.is_alive():
            LOGGER.warning("queue worker still alive after shutdown request")
        else:
            LOGGER.info("queue worker stopped")

    def _clone_record(self, rec: JobRecord) -> JobRecord:
        return JobRecord(**asdict(rec))

    def active_job_id(self) -> str | None:
        with self._lock:
            return self._active_job_id

    def _set_active_job_id(self, job_id: str | None) -> None:
        with self._lock:
            self._active_job_id = job_id

    def _get_active_process(self) -> subprocess.Popen | None:
        with self._lock:
            return self._active_process

    def _set_active_process(self, proc: subprocess.Popen | None) -> None:
        with self._lock:
            self._active_process = proc

    def submit(self, req: MeetingRunRequest) -> JobRecord:
        job_id = datetime.now().strftime("job_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
        work_dir = self.jobs_root / job_id
        work_dir.mkdir(parents=True, exist_ok=True)

        transcript_path = work_dir / "transcript.json"
        config_path = work_dir / "config.json"
        ocr_path = work_dir / "capture_ocr_results.json"
        html_path = work_dir / "meeting_report.html"
        log_path = work_dir / "orchestrator.log"

        mode = req.mode or ("agenda" if str(req.AGENDA_TEXT or "").strip() else "auto")
        report_layout = req.report_layout
        resume_artifact_dir = str(req.resume_artifact_dir or "").strip()

        transcript_payload = {
            "segments": req.segments,
            "full_text": req.full_text,
        }
        _safe_json_dump(transcript_path, transcript_payload)

        config_payload: dict[str, Any] = {"MEETING_INFO": req.MEETING_INFO}
        if str(req.AGENDA_TEXT or "").strip():
            config_payload["AGENDA_TEXT"] = req.AGENDA_TEXT
        if isinstance(req.TOPIC_TIME_OVERRIDES, list) and req.TOPIC_TIME_OVERRIDES:
            config_payload["TOPIC_TIME_OVERRIDES"] = req.TOPIC_TIME_OVERRIDES
        _safe_json_dump(config_path, config_payload)

        ocr_payload = dict(req.capture_ocr_results or {})
        if req.captures is not None:
            ocr_payload["captures"] = req.captures
        captures_raw = ocr_payload.get("captures", [])
        if not isinstance(captures_raw, list):
            captures_raw = []
        ocr_payload["captures"] = _normalize_captures(captures_raw)
        _safe_json_dump(ocr_path, ocr_payload)

        record = JobRecord(
            job_id=job_id,
            status="queued",
            created_at=_utc_now_iso(),
            started_at=None,
            finished_at=None,
            mode=mode,
            report_layout=report_layout,
            image_insert_enabled=bool(req.image_insert_enabled),
            save_artifacts=bool(req.save_artifacts),
            resume_artifact_dir=resume_artifact_dir,
            work_dir=str(work_dir),
            transcript_path=str(transcript_path),
            config_path=str(config_path),
            ocr_path=str(ocr_path),
            html_path=str(html_path),
            log_path=str(log_path),
            artifact_dir=None,
            runtime_log_path=None,
            error=None,
        )

        self._write_job_state(record)
        with self._lock:
            self._jobs[job_id] = record
            self._queue.put(job_id)
        LOGGER.info("job queued job_id=%s queue_size=%s", job_id, self._queue.qsize())
        return self._clone_record(record)

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            rec = self._jobs.get(job_id)
            return self._clone_record(rec) if rec else None

    def queue_position(self, job_id: str) -> int | None:
        rec = self.get(job_id)
        if rec is None:
            return None
        if rec.status == "running":
            return 0
        if rec.status != "queued":
            return None
        with self._queue.mutex:
            pending = [x for x in list(self._queue.queue) if x]
        try:
            return pending.index(job_id) + 1
        except ValueError:
            return None

    def _job_state_payload(self, record: JobRecord) -> dict[str, Any]:
        data = asdict(record)
        data["updated_at"] = _utc_now_iso()
        return data

    def _write_job_state(self, record: JobRecord) -> None:
        state_path = Path(record.work_dir) / "job_state.json"
        _safe_json_dump(state_path, self._job_state_payload(record))

    def _update(self, job_id: str, **fields: Any) -> JobRecord:
        with self._lock:
            rec = self._jobs[job_id]
            updated = replace(rec, **fields)
            self._jobs[job_id] = updated
        self._write_job_state(updated)
        return self._clone_record(updated)

    def _build_command(self, record: JobRecord) -> tuple[list[str], dict[str, str]]:
        cmd = [
            sys.executable,
            "orchestrator.py",
            "--mode",
            record.mode,
            "--report-layout",
            record.report_layout,
            "--output",
            record.html_path,
            "--save-artifacts",
            "true" if record.save_artifacts else "false",
        ]
        if record.resume_artifact_dir:
            cmd.extend(["--resume-artifact-dir", record.resume_artifact_dir])

        has_captures = False
        try:
            ocr_obj = json.loads(Path(record.ocr_path).read_text(encoding="utf-8"))
            captures = ocr_obj.get("captures", []) if isinstance(ocr_obj, dict) else []
            has_captures = isinstance(captures, list) and len(captures) > 0
        except Exception:
            has_captures = False

        env = os.environ.copy()
        env.update(
            {
                "TRANSCRIPT_PATH": record.transcript_path,
                "CONFIG_PATH": record.config_path,
                "OCR_PATH": record.ocr_path,
                "INCLUDE_OCR": "true" if has_captures else "false",
                "IMAGE_INSERT_ENABLED": "true" if record.image_insert_enabled else "false",
                "OUTPUT_HTML_PATH": record.html_path,
                # Queue-level single execution + pipeline-level single worker.
                "PIPELINE_MAX_CONCURRENCY": "1",
                # Presigned URLs usually don't have local file size metadata.
                "IMAGE_MIN_FILE_SIZE_KB": "0",
            }
        )
        return cmd, env

    def _find_latest_artifact_dir(self, work_dir: Path) -> Path | None:
        artifacts_root = work_dir / "artifacts"
        if not artifacts_root.exists():
            return None
        runs = [p for p in artifacts_root.iterdir() if p.is_dir()]
        if not runs:
            return None
        runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return runs[0]

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                job_id = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if job_id is None:
                self._queue.task_done()
                break
            try:
                self._run_one(job_id)
            finally:
                self._queue.task_done()

    def _run_one(self, job_id: str) -> None:
        rec = self.get(job_id)
        if rec is None:
            return
        self._set_active_job_id(job_id)
        self._update(
            job_id,
            status="running",
            started_at=_utc_now_iso(),
            finished_at=None,
            error=None,
        )
        LOGGER.info("job started job_id=%s", job_id)
        rec = self.get(job_id)
        assert rec is not None

        cmd, env = self._build_command(rec)
        log_path = Path(rec.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with log_path.open("w", encoding="utf-8") as logf:
                logf.write("$ " + " ".join(cmd) + "\n")
                logf.flush()
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(self.project_root),
                    env=env,
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                self._set_active_process(proc)
                exit_code = proc.wait()
                self._set_active_process(None)

            latest_artifact = self._find_latest_artifact_dir(Path(rec.work_dir))
            runtime_log = latest_artifact / "runtime.log" if latest_artifact else None
            html_exists = Path(rec.html_path).exists()

            if exit_code == 0 and html_exists:
                self._update(
                    job_id,
                    status="succeeded",
                    finished_at=_utc_now_iso(),
                    artifact_dir=str(latest_artifact) if latest_artifact else None,
                    runtime_log_path=str(runtime_log) if runtime_log and runtime_log.exists() else None,
                    error=None,
                )
                LOGGER.info("job succeeded job_id=%s", job_id)
                return

            tail = _tail_text(log_path, max_lines=80)
            err = f"orchestrator exited with code {exit_code}"
            if tail.strip():
                err = err + "\n" + tail
            self._update(
                job_id,
                status="failed",
                finished_at=_utc_now_iso(),
                artifact_dir=str(latest_artifact) if latest_artifact else None,
                runtime_log_path=str(runtime_log) if runtime_log and runtime_log.exists() else None,
                error=err,
            )
            LOGGER.error("job failed job_id=%s exit_code=%s", job_id, exit_code)
        except Exception as exc:
            self._update(
                job_id,
                status="failed",
                finished_at=_utc_now_iso(),
                error=f"job runner exception: {exc}",
            )
            LOGGER.exception("job runner exception job_id=%s", job_id)
        finally:
            self._set_active_process(None)
            self._set_active_job_id(None)


def _job_urls(request: Request, job_id: str) -> tuple[str, str, str]:
    status_url = str(request.url_for("get_job_status", job_id=job_id))
    html_url = str(request.url_for("get_job_html", job_id=job_id))
    logs_url = str(request.url_for("get_job_logs", job_id=job_id))
    return status_url, html_url, logs_url


def _read_result_html(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        LOGGER.warning("html decode warning path=%s (retry with ignore)", path)
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            LOGGER.exception("failed reading html path=%s", path)
            return None
    except Exception:
        LOGGER.exception("failed reading html path=%s", path)
        return None


def _build_job_create_response(payload: MeetingRunRequest, request: Request) -> JobCreateResponse:
    if not payload.segments:
        raise HTTPException(status_code=400, detail="segments is required and cannot be empty")
    if not str(payload.MEETING_INFO or "").strip():
        raise HTTPException(status_code=400, detail="MEETING_INFO is required")

    rec = MANAGER.submit(payload)
    queue_position = MANAGER.queue_position(rec.job_id)
    LOGGER.info("job accepted job_id=%s queue_position=%s", rec.job_id, queue_position)
    status_url, html_url, logs_url = _job_urls(request, rec.job_id)
    return JobCreateResponse(
        job_id=rec.job_id,
        status=rec.status,
        queue_position=queue_position,
        status_url=status_url,
        html_url=html_url,
        logs_url=logs_url,
    )


def _build_full_text_from_segments(segments: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for row in segments:
        if not isinstance(row, dict):
            continue
        text = str(row.get("text", "") or "").strip()
        if text:
            lines.append(text)
    return "\n".join(lines)


def _extract_transcript_payload(obj: Any) -> tuple[list[dict[str, Any]], str]:
    if isinstance(obj, dict):
        segments = obj.get("segments")
        full_text = obj.get("full_text")
    elif isinstance(obj, list):
        segments = obj
        full_text = None
    else:
        raise HTTPException(
            status_code=400,
            detail="file must be a JSON object or array with transcript segments",
        )

    if not isinstance(segments, list):
        raise HTTPException(status_code=400, detail="file.segments must be an array")

    for i, segment in enumerate(segments, start=1):
        if not isinstance(segment, dict):
            raise HTTPException(status_code=400, detail=f"file.segments[{i}] must be an object")

    if full_text is None:
        full_text_value = _build_full_text_from_segments(segments)
    else:
        full_text_value = str(full_text or "")

    return segments, full_text_value


async def _read_json_upload(upload: UploadFile, field_name: str) -> Any:
    raw = await upload.read()
    if not raw:
        raise HTTPException(status_code=400, detail=f"{field_name} is empty")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"{field_name} must be UTF-8 JSON") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"{field_name} is not valid JSON: {exc.msg}") from exc


def _parse_topic_time_overrides(raw: str | None) -> list[dict[str, Any]] | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"topic_time_overrides must be valid JSON: {exc.msg}") from exc
    if not isinstance(obj, list):
        raise HTTPException(status_code=400, detail="topic_time_overrides must be a JSON array")
    rows: list[dict[str, Any]] = []
    for i, row in enumerate(obj, start=1):
        if not isinstance(row, dict):
            raise HTTPException(status_code=400, detail=f"topic_time_overrides[{i}] must be an object")
        rows.append(row)
    return rows


async def _create_multipart_job(
    *,
    request: Request,
    attendees_text: str,
    agenda_text: str | None,
    transcript_file: UploadFile,
    ocr_file: UploadFile | None,
    default_report_layout: Literal["current", "react_official"],
    report_layout: Literal["current", "react_official"] | None,
    mode: Literal["agenda", "auto"] | None,
    topic_time_overrides: str | None,
) -> JobCreateResponse:
    transcript_obj = await _read_json_upload(transcript_file, "file")
    segments, full_text = _extract_transcript_payload(transcript_obj)

    capture_payload: dict[str, Any] | None = None
    if ocr_file is not None:
        ocr_obj = await _read_json_upload(ocr_file, "ocr_file")
        if isinstance(ocr_obj, dict):
            capture_payload = ocr_obj
        elif isinstance(ocr_obj, list):
            capture_payload = {"captures": ocr_obj}
        else:
            raise HTTPException(status_code=400, detail="ocr_file must be a JSON object or array")

    payload_dict: dict[str, Any] = {
        "MEETING_INFO": str(attendees_text or ""),
        "AGENDA_TEXT": str(agenda_text or ""),
        "segments": segments,
        "full_text": full_text,
        "capture_ocr_results": capture_payload,
        "report_layout": report_layout or default_report_layout,
        "mode": mode,
    }

    overrides = _parse_topic_time_overrides(topic_time_overrides)
    if overrides:
        payload_dict["TOPIC_TIME_OVERRIDES"] = overrides

    try:
        payload = MeetingRunRequest.model_validate(payload_dict)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc

    return _build_job_create_response(payload, request)


PROJECT_ROOT = Path(__file__).resolve().parent
JOBS_ROOT = Path(API_JOBS_ROOT).expanduser() if API_JOBS_ROOT else None
MANAGER = JobQueueManager(PROJECT_ROOT, jobs_root=JOBS_ROOT)

app = FastAPI(
    title="Meeting Summarizer Queue API",
    version="1.0.0",
    description=(
        "Queue API for orchestrator.py. "
        "Accepts meeting payload, runs one job at a time, returns HTML report output."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=API_CORS_ALLOW_ORIGINS,
    allow_credentials=API_CORS_ALLOW_CREDENTIALS,
    allow_methods=API_CORS_ALLOW_METHODS,
    allow_headers=API_CORS_ALLOW_HEADERS,
)


@app.middleware("http")
async def enforce_request_size_limit(request: Request, call_next):
    if request.method in {"POST", "PUT", "PATCH"}:
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > API_MAX_REQUEST_BODY_BYTES:
                    LOGGER.warning("request rejected: content-length too large path=%s", request.url.path)
                    return JSONResponse(
                        status_code=413,
                        content={"detail": f"request body too large (max {API_MAX_REQUEST_BODY_BYTES} bytes)"},
                    )
            except ValueError:
                return JSONResponse(status_code=400, content={"detail": "invalid Content-Length header"})

        body = await request.body()
        if len(body) > API_MAX_REQUEST_BODY_BYTES:
            LOGGER.warning("request rejected: body too large path=%s", request.url.path)
            return JSONResponse(
                status_code=413,
                content={"detail": f"request body too large (max {API_MAX_REQUEST_BODY_BYTES} bytes)"},
            )
    return await call_next(request)


@app.on_event("startup")
def on_startup() -> None:
    LOGGER.info("api startup")
    MANAGER.start()


@app.on_event("shutdown")
def on_shutdown() -> None:
    LOGGER.info("api shutdown")
    MANAGER.stop()


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "active_job_id": MANAGER.active_job_id()}


@app.post("/jobs", response_model=JobCreateResponse, status_code=202)
def create_job(payload: MeetingRunRequest, request: Request) -> JobCreateResponse:
    return _build_job_create_response(payload, request)


@app.post("/generate", response_model=JobCreateResponse, status_code=202)
async def generate_multipart(
    request: Request,
    attendees_text: str = Form(...),
    agenda_text: str | None = Form(default=None),
    file: UploadFile = File(...),
    ocr_file: UploadFile | None = File(default=None),
    topic_time_overrides: str | None = Form(default=None),
    TOPIC_TIME_OVERRIDES: str | None = Form(default=None),
    mode: Literal["agenda", "auto"] | None = Form(default=None),
    report_layout: Literal["current", "react_official"] | None = Form(default=None),
) -> JobCreateResponse:
    effective_overrides = TOPIC_TIME_OVERRIDES or topic_time_overrides
    return await _create_multipart_job(
        request=request,
        attendees_text=attendees_text,
        agenda_text=agenda_text,
        transcript_file=file,
        ocr_file=ocr_file,
        default_report_layout="react_official",
        report_layout=report_layout,
        mode=mode,
        topic_time_overrides=effective_overrides,
    )


@app.post("/generate_react", response_model=JobCreateResponse, status_code=202)
async def generate_react_multipart(
    request: Request,
    attendees_text: str = Form(...),
    agenda_text: str | None = Form(default=None),
    file: UploadFile = File(...),
    ocr_file: UploadFile | None = File(default=None),
    topic_time_overrides: str | None = Form(default=None),
    TOPIC_TIME_OVERRIDES: str | None = Form(default=None),
    mode: Literal["agenda", "auto"] | None = Form(default=None),
    report_layout: Literal["current", "react_official"] | None = Form(default=None),
) -> JobCreateResponse:
    effective_overrides = TOPIC_TIME_OVERRIDES or topic_time_overrides
    return await _create_multipart_job(
        request=request,
        attendees_text=attendees_text,
        agenda_text=agenda_text,
        transcript_file=file,
        ocr_file=ocr_file,
        default_report_layout="react_official",
        report_layout=report_layout,
        mode=mode,
        topic_time_overrides=effective_overrides,
    )


@app.get("/jobs/{job_id}", response_model=JobStatusResponse, name="get_job_status")
def get_job_status(job_id: str, request: Request) -> JobStatusResponse:
    rec = MANAGER.get(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="job not found")
    status_url, html_url, logs_url = _job_urls(request, job_id)
    return JobStatusResponse(
        job_id=rec.job_id,
        status=rec.status,
        queue_position=MANAGER.queue_position(job_id),
        created_at=rec.created_at,
        started_at=rec.started_at,
        finished_at=rec.finished_at,
        mode=rec.mode,
        report_layout=rec.report_layout,
        error=rec.error,
        work_dir=rec.work_dir,
        html_path=rec.html_path,
        log_path=rec.log_path,
        artifact_dir=rec.artifact_dir,
        runtime_log_path=rec.runtime_log_path,
        status_url=status_url,
        html_url=html_url,
        logs_url=logs_url,
    )


@app.get("/jobs/{job_id}/html", name="get_job_html")
def get_job_html(job_id: str) -> FileResponse:
    rec = MANAGER.get(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="job not found")
    if rec.status != "succeeded":
        raise HTTPException(status_code=409, detail=f"job status is {rec.status}, html is not ready")
    path = Path(rec.html_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="html output not found")
    return FileResponse(path=str(path), media_type="text/html", filename=path.name)


@app.get("/jobs/{job_id}/logs", name="get_job_logs")
def get_job_logs(
    job_id: str,
    tail: int = Query(default=200, ge=1, le=5000),
) -> PlainTextResponse:
    rec = MANAGER.get(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="job not found")
    log_path = Path(rec.log_path)
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="log not found")
    return PlainTextResponse(_tail_text(log_path, max_lines=tail))


@app.get("/jobs/{job_id}/result")
def get_job_result(job_id: str, request: Request) -> JSONResponse:
    rec = MANAGER.get(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="job not found")
    status_url, html_url, logs_url = _job_urls(request, job_id)
    html_path = Path(rec.html_path)
    html_ready = rec.status == "succeeded" and html_path.exists()
    minutes_html = _read_result_html(html_path) if html_ready else None
    payload: dict[str, Any] = {
        "job_id": rec.job_id,
        "status": rec.status,
        "status_url": status_url,
        "html_url": html_url,
        "logs_url": logs_url,
        "html_ready": html_ready,
        "error": rec.error,
        "artifact_dir": rec.artifact_dir,
        "runtime_log_path": rec.runtime_log_path,
    }
    if isinstance(minutes_html, str):
        payload["minutes_html"] = minutes_html
    return JSONResponse(payload)
