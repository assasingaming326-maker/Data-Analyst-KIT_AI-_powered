"""The plan contract (v1). Validation is strict: unknown transforms/types are rejected."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION = "1"
TYPES = {"string", "integer", "float", "boolean", "datetime", "category"}
TRANSFORMS = {"trim", "lower", "upper", "normalize_text", "strip_currency",
              "parse_percent", "split_name"}


class PlanError(ValueError):
    pass


@dataclass
class ColumnPlan:
    source: str
    target: str
    type: str
    nullable: bool = True
    transform: str | None = None
    format: str | None = None
    confidence: float = 1.0


@dataclass
class Plan:
    columns: list[ColumnPlan]
    category: str = "Unknown"
    drop: list[str] = field(default_factory=list)
    notes: str = ""
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def low_confidence(self, threshold: float) -> list[str]:
        return [c.source for c in self.columns if c.confidence < threshold]


_ALLOWED_COL_KEYS = {"source", "target", "type", "nullable", "transform", "format", "confidence"}


def validate_plan(raw: Any, known_columns: list[str] | None = None) -> Plan:
    """Validate untrusted dict (e.g. LLM output) into a Plan or raise PlanError."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as e:
            raise PlanError(f"not valid JSON: {e}") from e
    if not isinstance(raw, dict):
        raise PlanError("plan must be a JSON object")
    if str(raw.get("schema_version")) != SCHEMA_VERSION:
        raise PlanError(f"schema_version must be '{SCHEMA_VERSION}'")
    cols_raw = raw.get("columns")
    if not isinstance(cols_raw, list) or not cols_raw:
        raise PlanError("'columns' must be a non-empty list")
    cols, targets = [], set()
    for i, c in enumerate(cols_raw):
        if not isinstance(c, dict):
            raise PlanError(f"columns[{i}] must be an object")
        extra = set(c) - _ALLOWED_COL_KEYS
        if extra:
            raise PlanError(f"columns[{i}] has unknown keys: {sorted(extra)}")
        for k in ("source", "target", "type"):
            if not isinstance(c.get(k), str) or not c[k].strip():
                raise PlanError(f"columns[{i}].{k} is required")
        if c["type"] not in TYPES:
            raise PlanError(f"columns[{i}].type '{c['type']}' not in {sorted(TYPES)}")
        t = c.get("transform")
        if t is not None and t not in TRANSFORMS:
            raise PlanError(f"columns[{i}].transform '{t}' not in allow-list {sorted(TRANSFORMS)}")
        conf = c.get("confidence", 1.0)
        if not isinstance(conf, (int, float)) or isinstance(conf, bool) or not 0 <= conf <= 1:
            raise PlanError(f"columns[{i}].confidence must be a number in [0,1]")
        fmt = c.get("format")
        if fmt is not None and (not isinstance(fmt, str) or len(fmt) > 40):
            raise PlanError(f"columns[{i}].format invalid")
        if c["target"] in targets:
            raise PlanError(f"duplicate target '{c['target']}'")
        targets.add(c["target"])
        if known_columns is not None and c["source"] not in known_columns:
            raise PlanError(f"columns[{i}].source '{c['source']}' not found in file")
        cols.append(ColumnPlan(c["source"], c["target"], c["type"], bool(c.get("nullable", True)),
                               t, fmt, float(conf)))
    drop = raw.get("drop", [])
    if not isinstance(drop, list) or not all(isinstance(d, str) for d in drop):
        raise PlanError("'drop' must be a list of strings")
    return Plan(cols, str(raw.get("category", "Unknown"))[:50], drop, str(raw.get("notes", ""))[:500])
