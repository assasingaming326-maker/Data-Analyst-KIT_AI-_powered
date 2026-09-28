"""Per-column statistics used to build the LLM prompt."""
from __future__ import annotations
import re
import pandas as pd

_NUM = re.compile(r"^[-+]?[\d,]*\.?\d+$")
_DATE = re.compile(r"^\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}")
_BOOL = {"true", "false", "yes", "no", "y", "n"}


def infer_type(values: pd.Series) -> str:
    vals = values.astype(str).str.strip()
    vals = vals[vals != ""]
    if vals.empty:
        return "string"
    if vals.str.lower().isin(_BOOL).all():
        return "boolean"
    if vals.map(lambda v: bool(_NUM.match(v.replace(" ", "")))).all():
        return "integer" if vals.str.replace(",", "").str.match(r"^[-+]?\d+$").all() else "float"
    if vals.map(lambda v: bool(_DATE.match(v))).mean() > 0.9:
        return "datetime"
    return "category" if vals.nunique() <= max(10, len(vals) * 0.05) and vals.nunique() < len(vals) else "string"


def profile(df: pd.DataFrame, n_samples: int = 5) -> list[dict]:
    out = []
    for col in df.columns:
        s = df[col].astype(str)
        blank = s.str.strip() == ""
        nb = s[~blank]
        out.append({
            "name": col,
            "null_rate": round(float(blank.mean()), 4),
            "distinct": int(nb.nunique()),
            "min": nb.min() if len(nb) else None,
            "max": nb.max() if len(nb) else None,
            "inferred_type": infer_type(s),
            "samples": [v[:60] for v in nb.drop_duplicates().head(n_samples)],
        })
    return out
