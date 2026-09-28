"""Format-aware local file connector and parsers for tabular and text sources."""
from __future__ import annotations

import csv
import io
import json
from email import policy
from email.parser import BytesParser
from pathlib import Path
from xml.etree import ElementTree

import chardet
import pandas as pd
import yaml
from pypdf import PdfReader

STRUCTURED = {".csv", ".tsv", ".xlsx", ".xls", ".parquet"}
SEMI_STRUCTURED = {".json", ".jsonl", ".ndjson", ".xml", ".yaml", ".yml", ".log"}
UNSTRUCTURED = {".txt", ".md", ".html", ".htm", ".pdf", ".eml"}
MULTIMODAL = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}
SUPPORTED = STRUCTURED | SEMI_STRUCTURED | UNSTRUCTURED


class IngestError(ValueError):
    pass


def detect_encoding(path: Path) -> str:
    raw = path.read_bytes()[:200_000]
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    return chardet.detect(raw).get("encoding") or "utf-8"


def _text(path: Path) -> str:
    return path.read_bytes().decode(detect_encoding(path), errors="replace")


def describe_source(path: str | Path) -> dict:
    """Classify the source family and report whether a local parser is available."""
    path = Path(path)
    ext = path.suffix.lower()
    category = (
        "structured" if ext in STRUCTURED else
        "semi-structured" if ext in SEMI_STRUCTURED else
        "unstructured" if ext in UNSTRUCTURED else
        "multimodal" if ext in MULTIMODAL else "unknown"
    )
    if ext == ".txt" and path.exists() and path.stat().st_size:
        sample = _text(path)[:20_000]
        try:
            delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
            rows = list(csv.reader(sample.splitlines()[:20], delimiter=delimiter))
            if len(rows) > 1 and len({len(row) for row in rows}) == 1 and len(rows[0]) > 1:
                category = "structured"
        except csv.Error:
            pass
    descriptions = {
        "structured": "Tabular data with a defined row and column layout.",
        "semi-structured": "Tagged or record-based data that needs flattening.",
        "unstructured": "Text/document input that needs content extraction.",
        "multimodal": "Audio input requiring transcription before dataset conversion.",
        "unknown": "No source parser is registered for this file extension.",
    }
    return {
        "category": category,
        "format": ext.lstrip(".").upper() or "UNKNOWN",
        "supported": ext in SUPPORTED,
        "handling": descriptions[category],
    }


def _read_delimited(path: Path) -> pd.DataFrame:
    text = _text(path)
    if not text.strip():
        raise IngestError("File is empty")
    try:
        sep = csv.Sniffer().sniff(text[:20_000], delimiters=",;\t|").delimiter
    except csv.Error:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        return pd.read_csv(io.StringIO(text), sep=sep, dtype=str, keep_default_na=False,
                           on_bad_lines="skip")
    except Exception as e:
        raise IngestError(f"Could not parse as delimited text: {e}") from e


def _records_frame(data) -> pd.DataFrame:
    if isinstance(data, dict):
        records = next((value for value in data.values()
                        if isinstance(value, list) and all(isinstance(item, dict) for item in value)), None)
        data = records if records is not None else [data]
    elif not isinstance(data, list):
        data = [{"value": data}]
    if not data:
        raise IngestError("File contains no records")
    return pd.json_normalize(data).fillna("").astype(str)


def _xml_value(element):
    value = {f"@{key}": item for key, item in element.attrib.items()}
    children = list(element)
    if not children:
        text = (element.text or "").strip()
        if value:
            if text:
                value["#text"] = text
            return value
        return text
    for child in children:
        key = child.tag.rsplit("}", 1)[-1]
        child_value = _xml_value(child)
        if key in value:
            if not isinstance(value[key], list):
                value[key] = [value[key]]
            value[key].append(child_value)
        else:
            value[key] = child_value
    return value


def _read_xml(path: Path) -> pd.DataFrame:
    root = ElementTree.parse(path).getroot()
    children = list(root)
    records = [root] if children and all(not list(child) for child in children) else children
    if not records:
        records = [root]
    return _records_frame([_xml_value(record) for record in records])


def _read_html(path: Path) -> pd.DataFrame:
    from dtx.core.webpage import extract_html

    extracted = extract_html(_text(path), path.name)
    return pd.json_normalize(extracted["records"]).fillna("").astype(str)


def _read_pdf(path: Path) -> pd.DataFrame:
    reader = PdfReader(str(path), strict=False)
    pages = []
    for index, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"page": index, "text": text})
    if not pages:
        raise IngestError("No selectable PDF text found; scanned PDFs need an OCR connector")
    return pd.DataFrame(pages)


def _read_email(path: Path) -> pd.DataFrame:
    message = BytesParser(policy=policy.default).parsebytes(path.read_bytes())
    bodies = []
    parts = message.walk() if message.is_multipart() else [message]
    for part in parts:
        if part.get_content_type() == "text/plain" and part.get_content_disposition() != "attachment":
            try:
                bodies.append(part.get_content())
            except (LookupError, UnicodeDecodeError):
                continue
    return pd.DataFrame([{
        "from": str(message.get("from", "")),
        "to": str(message.get("to", "")),
        "subject": str(message.get("subject", "")),
        "date": str(message.get("date", "")),
        "body": "\n".join(bodies).strip(),
    }])


def read_file(path: str | Path) -> pd.DataFrame:
    """Extract a supported local source into a string DataFrame for profiling and planning."""
    path = Path(path)
    ext = path.suffix.lower()
    source = describe_source(path)
    if source["category"] == "multimodal":
        raise IngestError(f"{source['format']} is recognized as multimodal but has no OCR/transcription parser yet")
    if ext not in SUPPORTED:
        raise IngestError(f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED)}")
    if not path.exists() or path.stat().st_size == 0:
        raise IngestError("File is missing or empty")
    try:
        if ext in (".csv", ".tsv") or (ext == ".txt" and source["category"] == "structured"):
            df = _read_delimited(path)
        elif ext == ".txt":
            df = pd.DataFrame({"text": [line.strip() for line in _text(path).splitlines() if line.strip()]})
        elif ext in (".json", ".jsonl", ".ndjson"):
            text = _text(path)
            data = ([json.loads(line) for line in text.splitlines() if line.strip()]
                    if ext in (".jsonl", ".ndjson") else json.loads(text))
            df = _records_frame(data)
        elif ext in (".yaml", ".yml"):
            df = _records_frame(yaml.safe_load(_text(path)))
        elif ext == ".xml":
            df = _read_xml(path)
        elif ext in (".html", ".htm"):
            df = _read_html(path)
        elif ext == ".pdf":
            df = _read_pdf(path)
        elif ext == ".eml":
            df = _read_email(path)
        elif ext in (".md", ".log"):
            df = pd.DataFrame({"text": [line.strip() for line in _text(path).splitlines() if line.strip()]})
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(path, dtype=str).fillna("")
        else:
            df = pd.read_parquet(path).fillna("").astype(str)
    except IngestError:
        raise
    except Exception as e:
        raise IngestError(f"File appears corrupt or unreadable: {e}") from e
    df.columns = [str(column).strip() for column in df.columns]
    df = df.loc[:, [column for column in df.columns if column]]
    if df.empty:
        raise IngestError("File contains no data rows")
    return df.fillna("").astype(str)
