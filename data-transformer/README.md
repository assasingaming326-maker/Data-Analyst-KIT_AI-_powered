# Raw Data to Dataset Converter AI

Turns messy tabular files and text documents into clean, typed, validated datasets. The LLM (Groq) only
**proposes a schema mapping** from a *redacted* profile; all data processing is deterministic code.

## Quick start
```bash
cp .env.example .env            # add GROQ_API_KEY (optional: without it a rule-based plan is used)
pip install -e ".[dev]"
uvicorn dtx.api.app:app --reload   # dashboard at http://localhost:8000
# or CLI
dtx plan data.csv -o plan.json     # review/edit plan.json
dtx run data.csv plan.json --format parquet
# or Docker
docker compose up --build
pytest --cov=dtx.core
```

## Sources and validation
Local ingestion supports CSV/TSV, Excel, Parquet, JSON/JSONL, XML, YAML, HTML tables/text, PDFs with selectable text, email, Markdown, logs, and plain text. Sources are classified as structured, semi-structured, or unstructured and the classification is attached to each plan. Images and video are intentionally not supported; scanned PDFs need an OCR connector.

The dashboard has File and Web page source modes. Paste a public HTTPS page link to download its HTML, extract the title, headings, readable paragraphs/list items, and labeled table rows, then stage those records for AI planning. Scripts, navigation, footnotes, and repeated content are removed. DNS resolution is checked and pinned to public addresses; redirects are revalidated, response size is capped at 10 MB, and internal/private network targets are rejected. Pages that require JavaScript rendering are not supported by the current static HTML parser; if a page cannot be fetched, download/save it in your browser and use **Upload saved HTML** to run the same extractor. Reports let you select CSV, JSON, or Parquet for the dataset download; the reviewed schema template is available separately as JSON.

Exact duplicate output rows are removed by default and counted in the quality report; set `DTX_DEDUPLICATE_ROWS=false` to retain them. Every run validates planned output columns, data types, and non-null constraints. Vector indexing is not enabled because this converter currently produces datasets rather than semantic-search collections.

## Implemented vs blueprint
| Blueprint item | Status |
|---|---|
| Ingest (encoding/delimiter detect), profile, PII redaction | Done |
| Plan contract v1, allow-listed transforms, strict validation, hash | Done |
| LLM gateway (Groq, timeout, jittered backoff), retry with error feedback | Done |
| Fallback rule-based plan when LLM is down | Done |
| Deterministic transformer, quality checks, CSV/JSON/Parquet export, run report | Done |
| FastAPI job API, error format, API-key auth, upload limits, dashboard (plan editor) | Done |
| Docker, compose, CI (ruff, pytest, pip-audit, gitleaks) | Done |
| Postgres/Alembic, Celery/Redis, S3, Polars, OpenTelemetry, rate limiting, retention job, saved templates | **Not yet** — MVP uses pandas, JSON files on disk and FastAPI BackgroundTasks behind the same interfaces |

## Security note
Never paste API keys into chats or commit them. Rotate any key that has been shared.
