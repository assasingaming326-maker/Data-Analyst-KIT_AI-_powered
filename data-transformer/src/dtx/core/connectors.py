"""Local-only source connectors that stage external data before normal planning."""
from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

import pandas as pd

from dtx.pipeline import export

MAX_SOURCE_BYTES = 50 * 1024 * 1024
API_FORMATS = {".csv", ".json", ".jsonl", ".ndjson"}


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        return None


def _validate_api_url(url: str) -> None:
    parsed = urlsplit(url)
    local_http = parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    if parsed.scheme != "https" and not local_http:
        raise ValueError("API sources must use HTTPS (HTTP is allowed for localhost only)")
    if not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("API URL must include a host and must not embed credentials")


def _api_extension(url: str, content_type: str, body: bytes) -> str:
    suffix = Path(urlsplit(url).path).suffix.lower()
    if suffix in API_FORMATS:
        return suffix
    mime = content_type.lower()
    if "json" in mime:
        return ".jsonl" if "ndjson" in mime else ".json"
    if "csv" in mime or mime.startswith("text/"):
        return ".csv"
    try:
        json.loads(body)
        return ".json"
    except (UnicodeDecodeError, json.JSONDecodeError):
        return ".csv"


def stage_api(url: str, output: str | Path, headers: dict[str, str] | None = None,
              timeout: float = 30) -> Path:
    """Fetch a JSON/CSV API response into a local staging file for normal file ingestion."""
    _validate_api_url(url)
    request = Request(url, headers=headers or {}, method="GET")
    opener = build_opener(_NoRedirect())
    try:
        with opener.open(request, timeout=timeout) as response:
            body = response.read(MAX_SOURCE_BYTES + 1)
            content_type = response.headers.get("Content-Type", "")
    except (HTTPError, URLError, TimeoutError) as exc:
        raise ValueError(f"API source request failed: {exc}") from exc
    if len(body) > MAX_SOURCE_BYTES:
        raise ValueError("API response exceeds the 50 MB staging limit")
    if not body.strip():
        raise ValueError("API returned an empty response")

    suffix = _api_extension(url, content_type, body)
    if suffix in {".json", ".jsonl", ".ndjson"}:
        text = body.decode("utf-8-sig")
        if suffix in {".jsonl", ".ndjson"}:
            for line in text.splitlines():
                if line.strip():
                    json.loads(line)
        else:
            json.loads(text)
    else:
        pd.read_csv(pd.io.common.BytesIO(body), nrows=1)

    destination = Path(output)
    if destination.suffix.lower() not in API_FORMATS:
        destination = destination.with_suffix(suffix)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(body)
    return destination


def stage_sql(database_url: str, query: str, output: str | Path,
              parameters: dict | None = None) -> Path:
    """Execute one parameterized SELECT query locally and stage rows as CSV/JSON/Parquet."""
    statement = query.strip()
    if not re.match(r"^SELECT\b", statement, re.IGNORECASE) or ";" in statement.rstrip(";"):
        raise ValueError("Database connector accepts one SELECT statement only")
    try:
        from sqlalchemy import create_engine, text
    except ImportError as exc:
        raise RuntimeError("SQL staging requires SQLAlchemy; install with pip install 'dtx[connectors]'") from exc

    engine = create_engine(database_url, pool_pre_ping=True)
    try:
        with engine.connect() as connection:
            frame = pd.read_sql_query(text(statement), connection, params=parameters or {})
    finally:
        engine.dispose()
    if frame.empty:
        raise ValueError("Database query returned no rows")

    destination = Path(output)
    format_name = destination.suffix.lower().lstrip(".")
    if format_name not in {"csv", "json", "parquet"}:
        format_name = "parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)
    return export(frame, destination.with_suffix(""), format_name)