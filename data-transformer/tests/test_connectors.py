import json

import pytest

from dtx.core import connectors
from dtx.core.ingestion import read_file


def test_stage_api_stages_json_and_passes_headers(tmp_path, monkeypatch):
    captured = {}
    payload = b'{"records":[{"id":1},{"id":2}]}'

    class Response:
        def __init__(self):
            self.headers = {"Content-Type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def read(self, limit):
            return payload

    class Opener:
        def open(self, request, timeout):
            captured["url"] = request.full_url
            captured["headers"] = request.header_items()
            captured["timeout"] = timeout
            return Response()

    monkeypatch.setattr(connectors, "build_opener", lambda *_: Opener())
    staged = connectors.stage_api("https://api.example.test/records", tmp_path / "records",
                                  {"Authorization": "Bearer test"}, timeout=7)

    assert staged.suffix == ".json"
    assert json.loads(staged.read_text()) == {"records": [{"id": 1}, {"id": 2}]}
    assert captured["url"] == "https://api.example.test/records"
    assert ("Authorization", "Bearer test") in captured["headers"]
    assert captured["timeout"] == 7


def test_stage_api_rejects_insecure_remote_url(tmp_path):
    with pytest.raises(ValueError, match="HTTPS"):
        connectors.stage_api("http://api.example.test/records", tmp_path / "records")


def test_stage_sql_is_parameterized_and_select_only(tmp_path):
    sqlalchemy = pytest.importorskip("sqlalchemy")
    database = tmp_path / "source.db"
    engine = sqlalchemy.create_engine(f"sqlite:///{database}")
    with engine.begin() as connection:
        connection.exec_driver_sql("CREATE TABLE customers (id INTEGER, name TEXT)")
        connection.exec_driver_sql("INSERT INTO customers VALUES (1, 'Ada'), (2, 'Lin')")
    engine.dispose()

    staged = connectors.stage_sql(f"sqlite:///{database}",
                                  "SELECT id, name FROM customers WHERE id >= :minimum",
                                  tmp_path / "staged.parquet", {"minimum": 2})
    assert read_file(staged).to_dict("records") == [{"id": "2", "name": "Lin"}]
    with pytest.raises(ValueError, match="SELECT"):
        connectors.stage_sql(f"sqlite:///{database}", "DELETE FROM customers",
                             tmp_path / "invalid.csv")