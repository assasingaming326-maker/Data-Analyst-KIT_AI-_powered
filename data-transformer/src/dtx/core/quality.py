"""Post-transform checks: reconciliation, coerced-to-null, duplicates, nullability."""
from __future__ import annotations

import pandas as pd


def check_quality(rows_in: int, out: pd.DataFrame, stats: dict[str, dict],
                  max_null_introduced_pct: float = 20.0,
                  duplicates_removed: int = 0,
                  schema_errors: list[str] | None = None) -> dict:
    warnings, errors, columns = [], [], {}
    schema_errors = schema_errors or []
    errors.extend(schema_errors)
    expected_rows = rows_in - duplicates_removed
    if len(out) != expected_rows:
        errors.append(f"Row count mismatch: {expected_rows} expected after deduplication, {len(out)} out")
    for col, s in stats.items():
        introduced = max(0, s["nulls_after"] - s["nulls_before"])
        pct = round(100 * introduced / rows_in, 2) if rows_in else 0.0
        columns[col] = {"nulls_introduced": introduced, "nulls_introduced_pct": pct,
                        "nulls_total": s["nulls_after"]}
        if pct > max_null_introduced_pct:
            errors.append(f"'{col}': {pct}% of values failed to convert")
        elif introduced:
            warnings.append(f"'{col}': {introduced} value(s) failed to convert and became null")
        if not s["nullable"] and s["nulls_after"]:
            warnings.append(f"'{col}' is non-nullable but has {s['nulls_after']} null(s)")
    dups = int(out.duplicated().sum()) if len(out) else 0
    if dups:
        warnings.append(f"{dups} duplicate row(s)")
    return {"status": "fail" if errors else "warn" if warnings else "pass",
            "rows_in": rows_in, "rows_out": len(out), "duplicates": dups,
            "duplicates_removed": duplicates_removed,
            "schema_validation": {"valid": not schema_errors, "errors": schema_errors},
            "columns": columns, "warnings": warnings, "errors": errors}
