"""Deterministic application of a validated plan. No model-written code is ever executed."""
from __future__ import annotations

import unicodedata

import pandas as pd

from dtx.core.plan import Plan

_TRUE = {"true", "yes", "y", "1", "t"}
_FALSE = {"false", "no", "n", "0", "f"}


def _blank_to_na(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.strip()
    return s.mask(s == "", pd.NA)


def _apply_transform(s: pd.Series, name: str | None) -> pd.Series:
    if name in (None, "split_name"):
        return s
    if name == "trim":
        return s.str.strip()
    if name == "lower":
        return s.str.lower()
    if name == "normalize_text":
        def normalize(value):
            if pd.isna(value):
                return pd.NA
            text = unicodedata.normalize("NFKC", str(value)).lower()
            cleaned = "".join(char if char.isalnum() or char.isspace() else " " for char in text)
            return " ".join(cleaned.split())

        return s.map(normalize).astype("string")
    if name == "upper":
        return s.str.upper()
    if name == "strip_currency":
        return s.str.replace(r"[^\d.,\-]", "", regex=True).str.replace(",", "", regex=False)
    if name == "parse_percent":
        num = pd.to_numeric(s.str.replace("%", "", regex=False).str.strip(), errors="coerce")
        return (num / 100).astype("string")
    raise ValueError(f"unknown transform {name}")


def _to_bool(v):
    if pd.isna(v):
        return pd.NA
    v = str(v).lower()
    return True if v in _TRUE else False if v in _FALSE else pd.NA


def _cast(s: pd.Series, typ: str, fmt: str | None) -> pd.Series:
    if typ == "string":
        return s
    if typ in ("integer", "float"):
        num = pd.to_numeric(s.str.replace(",", "", regex=False), errors="coerce")
        return num.round().astype("Int64") if typ == "integer" else num.astype("Float64")
    if typ == "boolean":
        return s.map(_to_bool).astype("boolean")
    if typ == "datetime":
        return pd.to_datetime(s, format=fmt, errors="coerce")
    if typ == "category":
        return s.astype("category")
    raise ValueError(typ)


def apply_plan(df: pd.DataFrame, plan: Plan) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Return (clean_df, per-column stats)."""
    out, stats = {}, {}
    for c in plan.columns:
        raw = _blank_to_na(df[c.source])
        before = int(raw.isna().sum())
        t = _apply_transform(raw, c.transform)
        if c.transform == "split_name":
            parts = t.str.split(r"\s+", n=1, expand=True)
            t = parts[0]
            out[f"{c.target}_last"] = parts[1] if parts.shape[1] > 1 else pd.Series(pd.NA, index=t.index, dtype="string")
        casted = _cast(t, c.type, c.format)
        out[c.target] = casted
        stats[c.target] = {"nulls_before": before, "nulls_after": int(casted.isna().sum()),
                           "nullable": c.nullable}
    return pd.DataFrame(out), stats


def validate_output_schema(out: pd.DataFrame, plan: Plan) -> list[str]:
    """Check that transformed columns match the plan's names, order, types, and nullability."""
    expected = []
    errors = []
    for column in plan.columns:
        if column.transform == "split_name":
            expected.append(f"{column.target}_last")
        expected.append(column.target)
    if list(out.columns) != expected:
        errors.append(f"Output columns differ from schema: expected {expected}, received {list(out.columns)}")
    for column in plan.columns:
        if column.target not in out:
            continue
        series = out[column.target]
        valid_type = {
            "string": isinstance(series.dtype, pd.StringDtype),
            "integer": pd.api.types.is_integer_dtype(series.dtype),
            "float": pd.api.types.is_float_dtype(series.dtype),
            "boolean": pd.api.types.is_bool_dtype(series.dtype),
            "datetime": pd.api.types.is_datetime64_any_dtype(series.dtype),
            "category": isinstance(series.dtype, pd.CategoricalDtype),
        }.get(column.type, False)
        if not valid_type:
            errors.append(f"'{column.target}' does not match declared type '{column.type}'")
        if not column.nullable and series.isna().any():
            errors.append(f"'{column.target}' is non-nullable but contains null values")
    return errors
