"""Orchestration shared by the CLI and API. Same file + same plan = same output."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pandas as pd

from dtx.config import Settings
from dtx.core.ingestion import describe_source, read_file
from dtx.core.plan import Plan, validate_plan
from dtx.core.planner import PROMPT_VERSION, plan_with_llm, rule_based_plan
from dtx.core.profiling import profile
from dtx.core.quality import check_quality
from dtx.core.redaction import redact_profile
from dtx.core.transformer import apply_plan, validate_output_schema
from dtx.llm.base import LLMClient


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def propose_plan(path: Path, settings: Settings, llm: LLMClient | None = None,
                 sensitive: set[str] | None = None) -> dict:
    df = read_file(path)
    prof = profile(df)
    meta = {"prompt_version": PROMPT_VERSION, "model": None, "used_llm": False,
            "redacted": True, "source": describe_source(path)}
    try:
        if llm is None:
            if not settings.groq_api_key:
                raise RuntimeError("no LLM configured")
            from dtx.llm.groq_client import GroqClient
            llm = GroqClient(settings.groq_api_key, settings.groq_model)
        plan = plan_with_llm(llm, redact_profile(prof, sensitive), list(df.columns))
        meta.update(model=llm.model, used_llm=True)
    except Exception as e:  # graceful degradation
        plan = rule_based_plan(prof)
        meta["fallback_reason"] = str(e)[:200]
    return {"plan": plan.to_dict(), "plan_hash": plan.hash(), "meta": meta,
            "needs_review": plan.low_confidence(settings.confidence_threshold),
            "rows": len(df), "columns": list(df.columns)}


def run_transform(path: Path, plan: Plan, settings: Settings) -> tuple[pd.DataFrame, dict]:
    t0 = time.time()
    df = read_file(path)
    out, stats = apply_plan(df, plan)
    schema_errors = validate_output_schema(out, plan)
    duplicates_removed = int(out.duplicated().sum()) if settings.deduplicate_rows else 0
    if settings.deduplicate_rows:
        out = out.drop_duplicates(ignore_index=True)
    report = check_quality(len(df), out, stats, settings.max_null_introduced_pct,
                           duplicates_removed, schema_errors)
    report.update(plan_hash=plan.hash(), input_hash=file_hash(path),
                  duration_s=round(time.time() - t0, 3))
    return out, report


def export(df: pd.DataFrame, path: Path, fmt: str) -> Path:
    path = path.with_suffix(f".{fmt}")
    if fmt == "csv":
        df.to_csv(path, index=False)
    elif fmt == "json":
        df.to_json(path, orient="records", date_format="iso", indent=2)
    elif fmt == "parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"unsupported format {fmt}")
    return path


def load_plan_file(p: Path) -> Plan:
    return validate_plan(json.loads(p.read_text()))
