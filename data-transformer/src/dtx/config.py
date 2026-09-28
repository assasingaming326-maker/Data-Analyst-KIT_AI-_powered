"""Settings from environment variables. Secrets are never hard-coded."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _connector_hosts() -> tuple[str, ...]:
    hosts = []
    for entry in os.getenv("DTX_CONNECTOR_ALLOWED_HOSTS", "").split(","):
        host = entry.split("#", 1)[0].strip().lower().rstrip(".")
        if host and host not in hosts:
            hosts.append(host)
    return tuple(hosts)


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DTX_DATA_DIR", "./data")))
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    groq_model: str = field(default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
    confidence_threshold: float = field(default_factory=lambda: float(os.getenv("DTX_CONFIDENCE_THRESHOLD", "0.7")))
    max_upload_mb: int = field(default_factory=lambda: int(os.getenv("DTX_MAX_UPLOAD_MB", "50")))
    retention_days: int = field(default_factory=lambda: int(os.getenv("DTX_RETENTION_DAYS", "7")))
    api_key: str = field(default_factory=lambda: os.getenv("DTX_API_KEY", ""))
    connector_allowed_hosts: tuple[str, ...] = field(default_factory=_connector_hosts)
    deduplicate_rows: bool = field(default_factory=lambda: os.getenv(
        "DTX_DEDUPLICATE_ROWS", "true").strip().lower() in {"1", "true", "yes", "on"})
    max_null_introduced_pct: float = 20.0  # quality gate: fail above this per column

    def require_llm_key(self) -> str:
        if not self.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is not set")
        return self.groq_api_key
