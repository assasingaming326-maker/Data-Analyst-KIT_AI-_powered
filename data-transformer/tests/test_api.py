import json
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import dtx.api.app as api


def test_project_lifecycle_and_job_cleanup(tmp_path, monkeypatch):
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    monkeypatch.setattr(api, "JOBS", jobs)
    monkeypatch.setattr(api, "PROJECTS", tmp_path / "projects.json")
    api.app.dependency_overrides[api.auth] = lambda: None

    try:
        with TestClient(api.app) as client:
            created_response = client.post("/v1/projects", json={"name": "Untitled project"})
            assert created_response.status_code == 201
            project = created_response.json()
            assert project["draft"] is True

            owned_job = jobs / "owned-job"
            owned_job.mkdir()
            (owned_job / "state.json").write_text(json.dumps({"project_id": project["id"]}))
            other_job = jobs / "other-job"
            other_job.mkdir()
            (other_job / "state.json").write_text(json.dumps({"project_id": "other"}))

            filtered = client.get("/v1/jobs", params={"project_id": project["id"]})
            assert [job["project_id"] for job in filtered.json()] == [project["id"]]

            saved = client.patch(
                f"/v1/projects/{project['id']}", json={"name": "Customer records"}
            )
            assert saved.json()["name"] == "Customer records"
            assert saved.json()["draft"] is False

            deleted = client.delete(f"/v1/projects/{project['id']}")
            assert deleted.status_code == 200
            assert not owned_job.exists()
            assert other_job.exists()
            assert client.get("/v1/projects").json() == []
    finally:
        api.app.dependency_overrides.pop(api.auth, None)


def test_api_source_job_is_allowlisted_and_does_not_store_header(tmp_path, monkeypatch):
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    monkeypatch.setattr(api, "JOBS", jobs)
    monkeypatch.setattr(api, "PROJECTS", tmp_path / "projects.json")
    monkeypatch.setattr(api, "settings", replace(
        api.settings, data_dir=tmp_path, connector_allowed_hosts=("api.example.test",)
    ))
    monkeypatch.setattr(api, "_plan_job", lambda _: None)
    api.app.dependency_overrides[api.auth] = lambda: None
    captured = {}

    def stage_api(url, output, headers):
        captured.update(url=url, headers=headers)
        staged = Path(output).with_suffix(".json")
        staged.write_text('[{"name":"Ada"}]')
        return staged

    monkeypatch.setattr(api, "stage_api", stage_api)
    try:
        with TestClient(api.app) as client:
            project = client.post("/v1/projects", json={"name": "API import"}).json()
            denied = client.post("/v1/jobs/from-api", json={
                "project_id": project["id"], "url": "https://other.example.test/data.json"
            })
            assert denied.status_code == 403

            created = client.post("/v1/jobs/from-api", json={
                "project_id": project["id"], "url": "https://api.example.test/data.json",
                "header_name": "Authorization", "header_value": "Bearer private-token",
            })
            assert created.status_code == 200
            job = client.get("/v1/jobs", params={"project_id": project["id"]}).json()[0]
            assert job["id"] == created.json()["job_id"]
            assert job["filename"] == "data.json"
            assert "private-token" not in json.dumps(job)
            assert captured["headers"] == {"Authorization": "Bearer private-token"}
    finally:
        api.app.dependency_overrides.pop(api.auth, None)


def test_sql_source_job_stages_read_only_query(tmp_path, monkeypatch):
    sqlalchemy = pytest.importorskip("sqlalchemy")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    jobs = data_dir / "jobs"
    jobs.mkdir()
    database = data_dir / "source.sqlite"
    engine = sqlalchemy.create_engine(f"sqlite:///{database}")
    with engine.begin() as connection:
        connection.exec_driver_sql("CREATE TABLE samples (id INTEGER, label TEXT)")
        connection.exec_driver_sql("INSERT INTO samples VALUES (1, 'alpha')")
    engine.dispose()

    monkeypatch.setattr(api, "JOBS", jobs)
    monkeypatch.setattr(api, "PROJECTS", data_dir / "projects.json")
    monkeypatch.setattr(api, "settings", replace(api.settings, data_dir=data_dir))
    monkeypatch.setattr(api, "_plan_job", lambda _: None)
    api.app.dependency_overrides[api.auth] = lambda: None
    try:
        with TestClient(api.app) as client:
            project = client.post("/v1/projects", json={"name": "SQL import"}).json()
            created = client.post("/v1/jobs/from-sql", json={
                "project_id": project["id"], "database_url": f"sqlite:///{database}",
                "query": "SELECT id, label FROM samples WHERE id >= :minimum",
                "parameters": {"minimum": 1},
            })
            assert created.status_code == 200
            job = client.get("/v1/jobs", params={"project_id": project["id"]}).json()[0]
            assert job["id"] == created.json()["job_id"]
            assert job["filename"] == "database-query.parquet"
    finally:
        api.app.dependency_overrides.pop(api.auth, None)


def test_dashboard_shows_source_modes_and_quality_results():
    with TestClient(api.app) as client:
        page = client.get("/").text
    assert 'data-source-mode="file"' in page
    assert 'data-source-mode="page"' in page
    assert 'data-source-mode="api"' not in page
    assert 'data-source-mode="database"' not in page
    assert 'id="webpage-url"' in page
    assert 'id="upload-html-button"' in page
    assert 'id="dataset-format"' in page
    assert 'id="download-dataset"' in page
    assert 'id="schema-result"' in page
    assert 'id="duplicates-result"' in page
    assert '.jpg' not in page and '.mp4' not in page


def test_webpage_job_stages_extracted_records_in_project(tmp_path, monkeypatch):
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    monkeypatch.setattr(api, "JOBS", jobs)
    monkeypatch.setattr(api, "PROJECTS", tmp_path / "projects.json")
    monkeypatch.setattr(api, "settings", replace(api.settings, data_dir=tmp_path))
    monkeypatch.setattr(api, "_plan_job", lambda _: None)
    monkeypatch.setattr(api, "download_webpage", lambda url: (
        url, "<html lang='en'><title>Example article</title><main><h1>Intro</h1>"
        "<p>A meaningful paragraph from the page.</p></main></html>"
    ))
    api.app.dependency_overrides[api.auth] = lambda: None
    try:
        with TestClient(api.app) as client:
            project = client.post("/v1/projects", json={"name": "Web source"}).json()
            response = client.post("/v1/jobs/from-webpage", json={
                "project_id": project["id"], "url": "https://example.org/article"
            })
            assert response.status_code == 200
            job_id = response.json()["job_id"]
            state = client.get(f"/v1/jobs/{job_id}").json()
            assert state["filename"] == "Example-article-webpage.json"
            assert state["project_id"] == project["id"]
            extracted = json.loads((jobs / job_id / "input.json").read_text())
            assert any(record["text"] == "A meaningful paragraph from the page."
                       for record in extracted["records"])
    finally:
        api.app.dependency_overrides.pop(api.auth, None)