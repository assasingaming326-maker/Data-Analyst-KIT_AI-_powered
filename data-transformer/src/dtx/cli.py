"""dtx plan data.csv -> plan.json ; dtx run data.csv plan.json --format parquet"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dtx.config import Settings
from dtx.core.connectors import stage_api, stage_sql
from dtx.pipeline import export, load_plan_file, propose_plan, run_transform


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser("dtx")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan"); p.add_argument("file"); p.add_argument("-o", "--out", default="plan.json")
    r = sub.add_parser("run"); r.add_argument("file"); r.add_argument("plan")
    r.add_argument("--format", default="csv", choices=["csv", "json", "parquet"])
    r.add_argument("-o", "--out", default="output")
    api = sub.add_parser("stage-api", help="Fetch a JSON/CSV API response into local staging")
    api.add_argument("url"); api.add_argument("-o", "--out", required=True)
    api.add_argument("--headers-file", help="JSON file containing request headers; keep secrets local")
    api.add_argument("--timeout", type=float, default=30)
    sql = sub.add_parser("stage-sql", help="Stage rows from a read-only SQL SELECT query")
    sql.add_argument("database_url"); sql.add_argument("--query", required=True)
    sql.add_argument("--parameters", default="{}", help="JSON object of bound query parameters")
    sql.add_argument("-o", "--out", required=True)
    a = ap.parse_args(argv)
    s = Settings()
    if a.cmd == "stage-api":
        headers = json.loads(Path(a.headers_file).read_text()) if a.headers_file else {}
        if not isinstance(headers, dict) or not all(isinstance(k, str) and isinstance(v, str)
                                                    for k, v in headers.items()):
            ap.error("--headers-file must contain a JSON object of string headers")
        dest = stage_api(a.url, a.out, headers, a.timeout)
        print(f"API response staged at {dest}")
        return 0
    if a.cmd == "stage-sql":
        params = json.loads(a.parameters)
        if not isinstance(params, dict):
            ap.error("--parameters must be a JSON object")
        dest = stage_sql(a.database_url, a.query, a.out, params)
        print(f"Database rows staged at {dest}")
        return 0
    if a.cmd == "plan":
        res = propose_plan(Path(a.file), s)
        Path(a.out).write_text(json.dumps(res["plan"], indent=2))
        print(f"Plan written to {a.out} (hash {res['plan_hash']}, LLM used: {res['meta']['used_llm']})")
        if res["needs_review"]:
            print("Review these low-confidence columns:", ", ".join(res["needs_review"]))
        return 0
    df, report = run_transform(Path(a.file), load_plan_file(Path(a.plan)), s)
    dest = export(df, Path(a.out), a.format)
    Path(a.out + ".report.json").write_text(json.dumps(report, indent=2))
    print(f"{report['status'].upper()}: {report['rows_in']} -> {report['rows_out']} rows, wrote {dest}")
    for w in report["warnings"] + report["errors"]:
        print(" -", w)
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    sys.exit(main())
