# Meeting Summarizer (LangChain + LangGraph)

Pipeline สร้างรายงานประชุม HTML จาก 3 input:
- `transcript_YYYY-MM-DD.json`
- `config_YYYY-MM-DD.json`
- `capture_ocr_results.json`

## Project Structure

```text
.
├── orchestrator.py        # thin entrypoint (CLI + load .env + run graph)
├── workflow_graph.py      # LangGraph StateGraph (Agent 1..5 nodes + routing)
├── pipeline_utils.py      # config/dataclass + shared helpers + Agent1 reduce
├── llm_client.py          # LangChain chat routing + JSON repair/retry + token handling
├── html_renderer.py       # HTML compliance check + deterministic fallback renderer
├── image_processor.py     # image path resolve/base64/grouping helpers
├── prompts.py             # all agent prompts + CSS/JS bundle
└── data/...               # inputs
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
cp .env.example .env
```

แก้ `.env` ให้ครบก่อนรัน โดยรองรับ 2 นโยบาย:
- `Option A (default)`: `ALLOW_CHAT_FALLBACK=false` → Typhoon-only สำหรับ chat
- `Option B`: `ALLOW_CHAT_FALLBACK=true` → ถ้า Typhoon ใช้ไม่ได้จะ fallback ไป backend ที่กำหนดใน `CHAT_FALLBACK_PROVIDER` (`ollama` หรือ `vllm`)
- ตั้ง provider เพิ่มเติม:
  - `CHAT_FALLBACK_PROVIDER=ollama|vllm`
  - `EMBEDDING_PROVIDER=ollama|vllm`
  - ถ้าใช้ `vllm` ให้ตั้ง `VLLM_BASE_URL`, `VLLM_CHAT_MODEL`, `VLLM_EMBED_MODEL` ด้วย
  - หมายเหตุ: บาง vLLM deployment ไม่เปิด `/v1/embeddings` ควรใช้ `EMBEDDING_PROVIDER=ollama`
- token handling: ถ้า Typhoon ชน token limit ระบบจะลด `TYPHOON_MAX_TOKENS` อัตโนมัติและ shrink prompt ก่อนตัดสินใจ fallback

## Run

```bash
python orchestrator.py
python orchestrator.py --mode agenda
python orchestrator.py --report-layout react_official --output ./output/meeting_report_official.html
python orchestrator.py --mode auto --output ./output/report_auto.html --save-artifacts false
python orchestrator.py --resume-artifact-dir ./output/artifacts/run_20260224_104605
```

`--resume-artifact-dir` จะโหลด `agent1_cleaned.json` จาก run เดิมและข้าม Agent1 เพื่อรันต่อที่ Agent2
และถ้าโฟลเดอร์เดียวกันมี `agent2_kg.json` ด้วย ระบบจะข้าม Agent2 ต่ออัตโนมัติ

ปรับ chunk size ได้จาก `.env`:
- `AGENT1_CHUNK_SIZE` (default `120`)
- `AGENT1_CHUNK_OVERLAP` (default `1`)
- `AGENT1_SUBCHUNK_ON_FAILURE` (default `true`)
- `AGENT1_SUBCHUNK_SIZE` (default `40`)
- `AGENT1_OCR_MAX_CAPTURES` (default `3`, จำกัดจำนวน OCR ต่อ Agent1 chunk)
- `AGENT1_OCR_SNIPPET_CHARS` (default `220`, ตัดข้อความ OCR ต่อรูป)
- `OLLAMA_NUM_PREDICT` (default `4096`, เพดานจำนวน token ต่อคำตอบเมื่อใช้ Ollama chat)
- `AGENT2_CHUNK_SIZE` (default `160`)
- `AGENT25_CHUNK_SIZE` (default `12`)
- `PIPELINE_MAX_CONCURRENCY` (default `1`, ตั้ง `2` เพื่อยิง LLM พร้อมกัน 2 งาน)
- `REPORT_LAYOUT_MODE` (`current` | `react_official`, default `current`)
  - `current`: โครงรายงาน + CSS ปัจจุบัน
  - `react_official`: โครงรายงานเหมือน `current` ทุกส่วน แต่ apply CSS theme แบบ official ทับตอนท้าย

## FastAPI Queue Service

รัน API (คิวงานทีละ 1 งาน):

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

ตัวแปร env ที่เพิ่มเพื่อ hardening API:
- `API_CORS_ALLOW_ORIGINS` (default `*`, comma-separated)
- `API_CORS_ALLOW_METHODS` (default `*`)
- `API_CORS_ALLOW_HEADERS` (default `*`)
- `API_CORS_ALLOW_CREDENTIALS` (default `false`)
- `API_MAX_REQUEST_BODY_BYTES` (default `5242880`)
- `API_MAX_SEGMENTS` (default `10000`)
- `API_MAX_FULL_TEXT_CHARS` (default `1500000`)
- `API_MAX_MEETING_INFO_CHARS` (default `200000`)
- `API_MAX_AGENDA_TEXT_CHARS` (default `500000`)
- `API_MAX_TOPIC_TIME_OVERRIDES` (default `2000`)
- `API_MAX_CAPTURES` (default `30000`)
- `API_WORKER_JOIN_TIMEOUT_SEC` (default `10`)
- `API_PROCESS_TERMINATE_GRACE_SEC` (default `5`)
- `API_LOG_LEVEL` (default `INFO`)
- `API_JOBS_ROOT` (override path ของ `output/api_jobs`)

### Run With Docker

```bash
docker build -t meet-sum-api:latest .
docker run --rm -p 8000:8000 \
  --env-file .env \
  -v "$(pwd)/output:/app/output" \
  -v "$(pwd)/data:/app/data" \
  meet-sum-api:latest
```

### Submit Job

`POST /jobs` รับข้อมูล:
- `MEETING_INFO` (required)
- `AGENDA_TEXT` (optional)
- `TOPIC_TIME_OVERRIDES` (optional)
- `segments` (required)
- `full_text` (required)
- `capture_ocr_results` (optional, รองรับรูปแบบ `captures[]` ที่ `image_path` เป็น presigned URL ได้)

ตัวอย่าง:

```bash
curl -X POST http://127.0.0.1:8000/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "MEETING_INFO":"รายชื่อผู้เข้าประชุม...",
    "AGENDA_TEXT":"วาระที่ 1 ...",
    "TOPIC_TIME_OVERRIDES":[{"topic":"วาระที่ 1","start_time":"00:00:00","end_time":"00:10:00"}],
    "segments":[{"speaker":"A","start":0,"end":15,"text":"เริ่มประชุม"}],
    "full_text":"เริ่มประชุม...",
    "capture_ocr_results":{
      "captures":[
        {
          "capture_index":1,
          "timestamp_sec":0,
          "timestamp_hms":"00:00:00",
          "image_path":"https://...presigned...",
          "ocr_text":"ข้อความ OCR"
        }
      ]
    }
  }'
```

รองรับ endpoint แบบ multipart เพื่อให้ใช้งานร่วมกับ `video_minutes_service.py` ได้ตรงรูปแบบเดิม:
- `POST /generate` (default `report_layout=react_official`)
- `POST /generate_react` (default `report_layout=react_official`)

Multipart fields ที่รองรับ:
- `attendees_text` (required, map ไป `MEETING_INFO`)
- `agenda_text` (optional, map ไป `AGENDA_TEXT`)
- `file` (required, JSON transcript)
- `ocr_file` (optional, JSON OCR payload)
- `TOPIC_TIME_OVERRIDES` (optional, JSON string array)
- `topic_time_overrides` (optional, backward-compatible alias)
- `mode` (optional: `agenda` | `auto`)
- `report_layout` (optional: `current` | `react_official`, override ค่า default ของ endpoint)

ตัวอย่าง:

```bash
curl -X POST http://127.0.0.1:8000/generate_react \
  -F 'attendees_text=Alice, Bob' \
  -F 'agenda_text=Sprint review + action items' \
  -F 'TOPIC_TIME_OVERRIDES=[{"topic":"วาระที่ 1","start_time":"00:00:00","end_time":"00:10:00"}]' \
  -F 'file=@./data/transcript_2025-01-04.json;type=application/json' \
  -F 'ocr_file=@./data/video_change_ocr/run_20260212_160538/capture_ocr_results.json;type=application/json'
```

### Check Status / Result

- `GET /jobs/{job_id}` ดูสถานะ (`queued/running/succeeded/failed`)
- `GET /jobs/{job_id}/logs` ดู log run
- `GET /jobs/{job_id}/html` ดาวน์โหลดผล HTML
- `GET /jobs/{job_id}/result` ดูสรุปสถานะพร้อม URL และ (เมื่อพร้อม) แนบ HTML ใน key `minutes_html`

Output ของแต่ละ job จะอยู่ใน `output/api_jobs/<job_id>/`

ยิงจากไฟล์จริงในโปรเจกต์ด้วยสคริปต์:

```bash
scripts/submit_job_from_files.sh \
  ./data/config_2025-01-04_with_agenda.json \
  ./data/transcript_2025-01-04.json \
  ./data/video_change_ocr/run_20260213_163003/capture_ocr_results.json
```

## Output

- ไฟล์หลัก: `output/meeting_report.html`
- ไฟล์ตรวจสอบย้อนหลัง: `output/artifacts/run_YYYYMMDD_HHMMSS/`
  - `runtime.log` (log ละเอียดระดับ chunk)
  - `agent1_cleaned.json`
  - `agent2_kg.json`
  - `agent3_topic_map.json`
  - `agent25_image_manifest.json`
  - `agent4_summaries.json`
  - `agent5_report.html`
  - `run_metadata.json`

## Neo4j Visualization

รัน Neo4j (ตั้งรหัสผ่านให้ชัดเจนก่อน):

```bash
docker run -d --name meetsum-neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/neo4j12345 \
  neo4j:5
```

import `agent2_kg.json` เข้า Neo4j:

```bash
pip install -r requirements.txt
python3 scripts/import_agent2_kg_to_neo4j.py \
  --kg-path ./output/artifacts/agent2_kg.json \
  --uri bolt://localhost:7687 \
  --user neo4j \
  --password neo4j12345
```

เปิด Browser ที่ `http://localhost:7474` แล้วลอง query:

```cypher
MATCH (n) RETURN n LIMIT 200;
```

```cypher
MATCH (t:Topic)-[r]-(x) RETURN t, r, x LIMIT 300;
```

## Notes

- ใช้ map-reduce chunking สำหรับ transcript/OCR ขนาดใหญ่
- Agent1 ส่ง OCR เข้า LLM แบบย่อ (`title/flags/ocr_text snippet`) เพื่อลด JSON fail จากตาราง OCR ขนาดใหญ่
- Agent1 จะคัด OCR เฉพาะรูปที่ใกล้ช่วงเวลา chunk ที่สุดตาม `AGENT1_OCR_MAX_CAPTURES`
- Agent1 แยกทางทำงาน: transcript ใช้ LLM แบบ transcript-only และ OCR ใช้ LLM แบบ ocr-only (แยก call แล้ว merge)
- ฝั่ง OCR-only จะตัดข้อความ OCR เป็นชิ้นละ 1000 ตัวอักษร overlap 10% ก่อนส่ง LLM
- มี JSON repair + retry (สูงสุด 3 ครั้ง)
- ถ้า HTML จาก Agent 5 ไม่ครบโครงสร้าง จะ fallback เป็น deterministic renderer
- กรณี OCR image path ไม่ตรง filesystem มี resolver fallback หลายชั้น
