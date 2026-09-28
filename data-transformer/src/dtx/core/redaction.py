"""Mask PII before any value reaches the LLM."""
from __future__ import annotations
import re

PATTERNS = [
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("AADHAAR", re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b")),
    ("PHONE", re.compile(r"(?<!\w)\+?\d{1,3}[ -]?\(?\d{3,5}\)?[ -]?\d{3}[ -]?\d{3,4}(?!\w)")),
]


def redact_value(v: str) -> str:
    for label, rx in PATTERNS:
        v = rx.sub(f"<{label}>", v)
    return v


def redact_profile(prof: list[dict], sensitive: set[str] | None = None) -> list[dict]:
    """Return a copy of the profile safe to send to an LLM."""
    sensitive = sensitive or set()
    safe = []
    for p in prof:
        q = dict(p)
        if p["name"] in sensitive:
            q.update(samples=["<REDACTED>"], min=None, max=None)
        else:
            q["samples"] = [redact_value(s) for s in p["samples"]]
            q["min"] = redact_value(p["min"]) if p["min"] else p["min"]
            q["max"] = redact_value(p["max"]) if p["max"] else p["max"]
        safe.append(q)
    return safe
