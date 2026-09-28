import json
from pathlib import Path

import pandas as pd
import pytest

from dtx.config import Settings
from dtx.core.ingestion import IngestError, describe_source, read_file
from dtx.core.plan import PlanError, validate_plan
from dtx.core.planner import plan_with_llm, rule_based_plan
from dtx.core.profiling import profile
from dtx.core.redaction import redact_profile, redact_value
from dtx.core.transformer import apply_plan, validate_output_schema
from dtx.llm.groq_client import GroqClient
from dtx.pipeline import propose_plan, run_transform

FIX = Path(__file__).parent / "fixtures" / "messy.csv"

GOOD = {"schema_version": "1", "category": "Sales", "columns": [
    {"source": "Cust Name", "target": "customer_name", "type": "string", "transform": "trim", "nullable": False, "confidence": 0.94},
    {"source": "Order Dt", "target": "order_date", "type": "datetime", "format": "%d/%m/%Y", "confidence": 0.88},
    {"source": "Amount", "target": "amount", "type": "float", "transform": "strip_currency", "confidence": 0.5},
], "drop": ["Unnamed: 4"], "notes": ""}


class FakeLLM:
    model = "fake"
    def __init__(self, replies): self.replies, self.calls = list(replies), []
    def complete_json(self, system, user):
        self.calls.append(user); return self.replies.pop(0)


def test_plan_rejects_hostile():
    bad = json.loads(json.dumps(GOOD)); bad["columns"][0]["transform"] = "__import__('os').system('x')"
    with pytest.raises(PlanError): validate_plan(bad)
    bad = json.loads(json.dumps(GOOD)); bad["columns"][0]["type"] = "blob"
    with pytest.raises(PlanError): validate_plan(bad)
    bad = json.loads(json.dumps(GOOD)); bad["columns"][0]["code"] = "x"
    with pytest.raises(PlanError): validate_plan(bad)
    with pytest.raises(PlanError): validate_plan("not json")
    with pytest.raises(PlanError): validate_plan(GOOD, known_columns=["nope"])


def test_plan_hash_stable():
    assert validate_plan(GOOD).hash() == validate_plan(json.loads(json.dumps(GOOD))).hash()


def test_ingest_and_errors(tmp_path):
    assert list(read_file(FIX).columns)[:2] == ["Cust Name", "Order Dt"]
    p = tmp_path / "a.exe"; p.write_text("x")
    with pytest.raises(IngestError): read_file(p)
    e = tmp_path / "e.csv"; e.write_text("")
    with pytest.raises(IngestError): read_file(e)
    j = tmp_path / "a.json"; j.write_text(json.dumps([{"a": 1, "b": None}]))
    assert read_file(j).shape == (1, 2)
    s = tmp_path / "s.csv"; s.write_bytes("name;city\nJosé;Zürich\n".encode("latin-1"))
    assert read_file(s).shape == (1, 2)


def test_ingestion_classifies_and_parses_source_families(tmp_path):
    jsonl = tmp_path / "events.jsonl"
    jsonl.write_text('{"event":"open","user":{"id":1}}\n{"event":"close","user":{"id":2}}\n')
    assert describe_source(jsonl)["category"] == "semi-structured"
    assert list(read_file(jsonl)["user.id"]) == ["1", "2"]

    xml = tmp_path / "records.xml"
    xml.write_text("<records><record><name>Ada</name><team>Data</team></record>"
                   "<record><name>Lin</name><team>Ops</team></record></records>")
    assert read_file(xml).to_dict("records") == [
        {"name": "Ada", "team": "Data"}, {"name": "Lin", "team": "Ops"}
    ]

    yaml_file = tmp_path / "records.yaml"
    yaml_file.write_text("records:\n  - id: 1\n    label: alpha\n  - id: 2\n    label: beta\n")
    assert list(read_file(yaml_file)["id"]) == ["1", "2"]

    html = tmp_path / "table.html"
    html.write_text("<html><script>private()</script><table><tr><th>code</th></tr>"
                    "<tr><td>A1</td></tr></table></html>")
    html_rows = read_file(html)
    assert html_rows.iloc[0]["text"] == "code: A1"
    assert "private" not in " ".join(html_rows["text"])

    parquet = tmp_path / "records.parquet"
    pd.DataFrame({"code": ["A1"], "count": [2]}).to_parquet(parquet)
    assert read_file(parquet).to_dict("records") == [{"code": "A1", "count": "2"}]

    email = tmp_path / "message.eml"
    email.write_text("From: analyst@example.test\nSubject: Daily totals\n"
                     "Content-Type: text/plain; charset=utf-8\n\nRevenue was 1200.\n")
    assert read_file(email).iloc[0]["subject"] == "Daily totals"
    assert "Revenue was 1200" in read_file(email).iloc[0]["body"]

    image = tmp_path / "scan.jpg"
    image.write_bytes(b"not decoded")
    assert describe_source(image)["category"] == "unknown"
    with pytest.raises(IngestError, match="Unsupported file type"):
        read_file(image)


def test_redaction():
    assert redact_value("mail me at a.b@x.com or 4111 1111 1111 1111") == "mail me at <EMAIL> or <CARD>"
    prof = redact_profile(profile(read_file(FIX)), sensitive={"Cust Name"})
    dump = json.dumps(prof)
    assert "@example.com" not in dump and "alice" not in dump.lower()


def test_transform_and_quality():
    df = read_file(FIX)
    out, stats = apply_plan(df, validate_plan(GOOD))
    assert list(out.columns) == ["customer_name", "order_date", "amount"]
    assert out["customer_name"][0] == "alice smith"
    assert out["amount"][0] == 1200.5 and pd.isna(out["amount"][2])
    assert pd.isna(out["order_date"][2]) and out["order_date"][1] == pd.Timestamp(2024, 3, 15)
    assert stats["order_date"]["nulls_after"] - stats["order_date"]["nulls_before"] == 1


def test_run_transform_reproducible():
    plan = validate_plan(GOOD)
    a, ra = run_transform(FIX, plan, Settings())
    b, rb = run_transform(FIX, plan, Settings())
    assert a.equals(b) and ra["rows_in"] == ra["rows_out"] == 3
    assert ra["status"] == "fail"  # 33% failed conversions exceeds 20% gate
    assert ra["plan_hash"] == rb["plan_hash"]


def test_schema_validation_and_duplicate_removal(tmp_path):
    source = tmp_path / "duplicate.csv"
    source.write_text("name\nAda\nAda\nLin\n")
    plan = validate_plan({"schema_version": "1", "category": "People", "columns": [
        {"source": "name", "target": "name", "type": "string", "nullable": False}
    ], "drop": []}, known_columns=["name"])

    transformed, _ = apply_plan(read_file(source), plan)
    assert validate_output_schema(transformed, plan) == []
    cleaned, report = run_transform(source, plan, Settings(deduplicate_rows=True))
    assert cleaned["name"].tolist() == ["Ada", "Lin"]
    assert report["duplicates_removed"] == 1
    assert report["rows_out"] == report["rows_in"] - 1
    assert report["schema_validation"]["valid"] is True


def test_normalize_text_transform():
    plan = validate_plan({"schema_version": "1", "columns": [
        {"source": "label", "target": "label", "type": "string",
         "transform": "normalize_text"}
    ]})
    frame, _ = apply_plan(pd.DataFrame({"label": ["  Café & Cake!!  "]}), plan)
    assert frame["label"].iloc[0] == "café cake"


def test_llm_retry_feeds_error_back():
    llm = FakeLLM(["{}", json.dumps(GOOD)])
    plan = plan_with_llm(llm, [], ["Cust Name", "Order Dt", "Amount"])
    assert len(plan.columns) == 3 and "invalid" in llm.calls[1]
    with pytest.raises(PlanError):
        plan_with_llm(FakeLLM(["{}"] * 3), [], ["a"])


def test_groq_client_uses_responses_api(monkeypatch):
    captured = {}

    class Responses:
        def create(self, **kwargs):
            captured["request"] = kwargs
            return type("Response", (), {"output_text": '{"ok": true}'})()

    class Client:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.responses = Responses()

    monkeypatch.setattr("dtx.llm.groq_client.OpenAI", Client)
    llm = GroqClient("test-key", "openai/gpt-oss-20b")

    assert llm.complete_json("system prompt", "user prompt") == '{"ok": true}'
    assert captured["client"]["base_url"] == "https://api.groq.com/openai/v1"
    assert captured["request"]["model"] == "openai/gpt-oss-20b"
    assert captured["request"]["instructions"] == "system prompt"
    assert captured["request"]["input"] == "user prompt"
    assert captured["request"]["text"] == {"format": {"type": "json_object"}}


def test_fallback_when_llm_down(tmp_path):
    class Down(FakeLLM):
        def complete_json(self, s, u): raise RuntimeError("down")
    res = propose_plan(FIX, Settings(), llm=Down([]))
    assert res["meta"]["used_llm"] is False and "Unnamed: 4" in res["plan"]["drop"]
    assert res["plan"]["columns"][0]["target"] == "cust_name"


def test_prompt_injection_cell_cannot_execute():
    evil = json.dumps({**GOOD, "columns": [{**GOOD["columns"][0], "transform": "exec"}]})
    with pytest.raises(PlanError):
        plan_with_llm(FakeLLM([evil] * 3), [], ["Cust Name"])


def test_rule_based_plan_valid():
    assert rule_based_plan(profile(read_file(FIX))).columns
