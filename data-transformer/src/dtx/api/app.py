"""FastAPI service. MVP job store = JSON files on disk (swap for Postgres + Celery/RQ later)."""
from __future__ import annotations

import json
import re
import shutil
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from dtx.config import Settings
from dtx.core.connectors import stage_api, stage_sql
from dtx.core.ingestion import SUPPORTED, describe_source
from dtx.core.plan import PlanError, validate_plan
from dtx.core.webpage import download_webpage, extract_html
from dtx.pipeline import export, propose_plan, run_transform

settings = Settings()
app = FastAPI(title="Raw Data to Dataset Converter AI", version="1.0")
JOBS = settings.data_dir / "jobs"
PROJECTS = settings.data_dir / "projects.json"
JOBS.mkdir(parents=True, exist_ok=True)


class ProjectInput(BaseModel):
    name: str = Field(min_length=1, max_length=80)


class ApiSourceInput(BaseModel):
    project_id: str = Field(min_length=12, max_length=12)
    url: str = Field(min_length=8, max_length=2048)
    header_name: str | None = Field(default=None, max_length=120)
    header_value: str | None = Field(default=None, max_length=4096)


class SqlSourceInput(BaseModel):
    project_id: str = Field(min_length=12, max_length=12)
    database_url: str = Field(min_length=8, max_length=4096)
    query: str = Field(min_length=6, max_length=20_000)
    parameters: dict = Field(default_factory=dict)


class WebpageSourceInput(BaseModel):
    project_id: str = Field(min_length=12, max_length=12)
    url: str = Field(min_length=12, max_length=2048)


def err(status: int, code: str, message: str, hint: str = ""):
    raise HTTPException(status, {"error": {"code": code, "message": message, "hint": hint}})


@app.exception_handler(HTTPException)
async def _http_error(_, exc: HTTPException):
    body = exc.detail if isinstance(exc.detail, dict) else {
        "error": {"code": "ERROR", "message": str(exc.detail), "hint": ""}}
    return JSONResponse(body, status_code=exc.status_code)


def auth(x_api_key: str = Header(default="")):
    if settings.api_key and x_api_key != settings.api_key:
        err(401, "UNAUTHORIZED", "Missing or invalid API key", "Send the X-API-Key header")


def jdir(job_id: str) -> Path:
    d = JOBS / job_id
    if not job_id.isalnum() or not d.exists():
        err(404, "JOB_NOT_FOUND", f"No job '{job_id}'")
    return d


def get_state(d: Path) -> dict:
    return json.loads((d / "state.json").read_text())


def set_state(d: Path, **kw):
    s = get_state(d) if (d / "state.json").exists() else {}
    s.update(kw)
    (d / "state.json").write_text(json.dumps(s, indent=2))


def get_projects() -> list[dict]:
    if not PROJECTS.exists():
        return []
    return json.loads(PROJECTS.read_text())


def save_projects(projects: list[dict]):
    PROJECTS.parent.mkdir(parents=True, exist_ok=True)
    temp = PROJECTS.with_suffix(".tmp")
    temp.write_text(json.dumps(projects, indent=2))
    temp.replace(PROJECTS)


def find_project(project_id: str) -> dict | None:
    return next((p for p in get_projects() if p["id"] == project_id), None)


def _plan_job(d: Path):
    try:
        set_state(d, status="planning")
        res = propose_plan(d / get_state(d)["stored_name"], settings)
        (d / "plan.json").write_text(json.dumps(res["plan"], indent=2))
        set_state(d, status="needs_review", plan_hash=res["plan_hash"], meta=res["meta"],
                  needs_review=res["needs_review"], rows=res["rows"])
    except Exception as e:
        set_state(d, status="failed", error=str(e))


def _run_job(d: Path):
    try:
        plan = validate_plan(json.loads((d / "plan.json").read_text()))
        df, report = run_transform(d / get_state(d)["stored_name"], plan, settings)
        for fmt in ("csv", "json", "parquet"):
            export(df, d / "output", fmt)
        (d / "report.json").write_text(json.dumps(report, indent=2))
        failed = report["status"] == "fail"
        set_state(d, status="failed" if failed else "done", error="; ".join(report["errors"]) or None)
    except Exception as e:
        set_state(d, status="failed", error=str(e))


def _queue_staged_job(project_id: str, path: Path, filename: str, bg: BackgroundTasks) -> dict:
    if not find_project(project_id):
        err(404, "PROJECT_NOT_FOUND", f"No project '{project_id}'")
    ext = path.suffix.lower()
    if ext not in SUPPORTED:
        err(415, "UNSUPPORTED_TYPE", f"Connector returned unsupported format '{ext}'")
    if path.stat().st_size > settings.max_upload_mb << 20:
        err(413, "TOO_LARGE", f"Source exceeds {settings.max_upload_mb} MB")
    job_id = uuid.uuid4().hex[:12]
    directory = JOBS / job_id
    directory.mkdir()
    stored = f"input{ext}"
    shutil.copyfile(path, directory / stored)
    set_state(directory, id=job_id, filename=Path(filename).name[:200], stored_name=stored,
              status="queued", created=time.time(), project_id=project_id)
    bg.add_task(_plan_job, directory)
    return {"job_id": job_id}


def _check_connector_host(host: str | None):
    if not host or host.lower().rstrip(".") not in settings.connector_allowed_hosts:
        err(403, "CONNECTOR_HOST_NOT_ALLOWED", "Connector host is not enabled",
            "Add the exact API/database host to DTX_CONNECTOR_ALLOWED_HOSTS in .env and restart")


@app.post("/v1/jobs", dependencies=[Depends(auth)])
async def create_job(
    bg: BackgroundTasks,
    file: UploadFile = File(...),
    project_id: str = Form(default=""),
):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in SUPPORTED:
        source = describe_source(file.filename or "")
        hint = (source["handling"] if source["category"] == "multimodal"
            else f"Use a supported local format: {sorted(SUPPORTED)}")
        err(415, "UNSUPPORTED_TYPE", f"'{ext}' is not supported", hint)
    if project_id and not find_project(project_id):
        err(404, "PROJECT_NOT_FOUND", f"No project '{project_id}'")
    job_id = uuid.uuid4().hex[:12]
    d = JOBS / job_id
    d.mkdir()
    stored, size = f"input{ext}", 0
    with open(d / stored, "wb") as f:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > settings.max_upload_mb << 20:
                f.close()
                shutil.rmtree(d)
                err(413, "TOO_LARGE", f"File exceeds {settings.max_upload_mb} MB")
            f.write(chunk)
    set_state(d, id=job_id, filename=file.filename, stored_name=stored, status="queued",
              created=time.time(), project_id=project_id or None)
    bg.add_task(_plan_job, d)
    return {"job_id": job_id}


@app.get("/v1/connectors", dependencies=[Depends(auth)])
def connector_capabilities():
    try:
        import sqlalchemy  # noqa: F401
        sql_enabled = True
    except ImportError:
        sql_enabled = False
    return {"api_enabled": True, "sql_enabled": sql_enabled,
            "allowed_hosts": list(settings.connector_allowed_hosts)}


@app.post("/v1/jobs/from-webpage", dependencies=[Depends(auth)])
def create_webpage_job(source: WebpageSourceInput, bg: BackgroundTasks):
    if not find_project(source.project_id):
        err(404, "PROJECT_NOT_FOUND", f"No project '{source.project_id}'")
    staging = settings.data_dir / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    temp_path = staging / f"web-{uuid.uuid4().hex}"
    html_path = temp_path.with_suffix(".html")
    records_path = temp_path.with_suffix(".json")
    try:
        final_url, markup = download_webpage(source.url)
        html_path.write_text(markup)
        extracted = extract_html(html_path.read_text(), final_url)
        records_path.write_text(json.dumps(extracted, ensure_ascii=False))
        page_title = re.sub(r"[^A-Za-z0-9._-]+", "-", extracted["title"]).strip("-._")[:100]
        filename = f"{page_title or urlsplit(source.url).hostname}-webpage.json"
        return _queue_staged_job(source.project_id, records_path, filename, bg)
    except ValueError as exc:
        err(400, "WEBPAGE_FETCH_FAILED", str(exc))
    except Exception:
        err(400, "WEBPAGE_FETCH_FAILED", "Could not extract this webpage")
    finally:
        html_path.unlink(missing_ok=True)
        records_path.unlink(missing_ok=True)


@app.post("/v1/jobs/from-api", dependencies=[Depends(auth)])
def create_api_job(source: ApiSourceInput, bg: BackgroundTasks):
    parsed = urlsplit(source.url)
    _check_connector_host(parsed.hostname)
    headers = {}
    if source.header_name or source.header_value:
        if not source.header_name or not source.header_value:
            err(422, "CONNECTOR_HEADER_INVALID", "Provide both an authorization header name and value")
        headers[source.header_name] = source.header_value
    staging = settings.data_dir / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    temp_path = staging / f"api-{uuid.uuid4().hex}"
    try:
        staged = stage_api(source.url, temp_path, headers)
        filename = Path(parsed.path).name or f"api-source{staged.suffix}"
        return _queue_staged_job(source.project_id, staged, filename, bg)
    except ValueError:
        err(400, "API_SOURCE_FAILED", "Could not stage the API response",
            "Check the HTTPS endpoint, authorization, host allowlist, and JSON/CSV response")
    finally:
        for candidate in staging.glob(f"{temp_path.name}*"):
            candidate.unlink(missing_ok=True)


@app.post("/v1/jobs/from-sql", dependencies=[Depends(auth)])
def create_sql_job(source: SqlSourceInput, bg: BackgroundTasks):
    try:
        from sqlalchemy.engine import make_url
    except ImportError:
        err(503, "SQL_CONNECTOR_UNAVAILABLE", "Database connector is not installed",
            "Install with pip install -e '.[connectors]' and restart")
    try:
        connection = make_url(source.database_url)
    except Exception:
        err(422, "DATABASE_URL_INVALID", "Database connection URL is invalid")
    if connection.drivername.startswith("sqlite"):
        database_path = Path(connection.database or "").expanduser().resolve()
        data_root = settings.data_dir.resolve()
        if data_root not in database_path.parents:
            err(403, "DATABASE_PATH_NOT_ALLOWED", "SQLite sources must be inside DTX_DATA_DIR")
    else:
        _check_connector_host(connection.host)

    staging = settings.data_dir / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    temp_path = staging / f"sql-{uuid.uuid4().hex}.parquet"
    try:
        staged = stage_sql(source.database_url, source.query, temp_path, source.parameters)
    except ValueError as exc:
        err(422, "SQL_SOURCE_INVALID", str(exc))
    except Exception:
        err(400, "SQL_SOURCE_FAILED", "Could not stage database rows",
            "Check the allowlisted host, driver, credentials, and SELECT query")
    try:
        return _queue_staged_job(source.project_id, staged, "database-query" + staged.suffix, bg)
    finally:
        staged.unlink(missing_ok=True)


@app.get("/v1/jobs", dependencies=[Depends(auth)])
def list_jobs(project_id: str | None = None):
    states = [get_state(p) for p in JOBS.iterdir() if (p / "state.json").exists()]
    if project_id:
        states = [s for s in states if s.get("project_id") == project_id]
    return sorted(states, key=lambda s: -s.get("created", 0))


@app.get("/v1/projects", dependencies=[Depends(auth)])
def list_projects():
    return sorted(get_projects(), key=lambda p: -p["updated"])


@app.post("/v1/projects", status_code=201, dependencies=[Depends(auth)])
def create_project(project: ProjectInput):
    name = project.name.strip()
    if not name:
        err(422, "PROJECT_NAME_REQUIRED", "Project name cannot be blank")
    now = time.time()
    created = {"id": uuid.uuid4().hex[:12], "name": name, "created": now,
               "updated": now, "draft": True}
    projects = get_projects()
    projects.append(created)
    save_projects(projects)
    return created


@app.patch("/v1/projects/{project_id}", dependencies=[Depends(auth)])
def update_project(project_id: str, project: ProjectInput):
    name = project.name.strip()
    if not name:
        err(422, "PROJECT_NAME_REQUIRED", "Project name cannot be blank")
    projects = get_projects()
    updated = next((p for p in projects if p["id"] == project_id), None)
    if updated is None:
        err(404, "PROJECT_NOT_FOUND", f"No project '{project_id}'")
    updated.update(name=name, updated=time.time(), draft=False)
    save_projects(projects)
    return updated


@app.delete("/v1/projects/{project_id}", dependencies=[Depends(auth)])
def delete_project(project_id: str):
    projects = get_projects()
    if not any(p["id"] == project_id for p in projects):
        err(404, "PROJECT_NOT_FOUND", f"No project '{project_id}'")
    for directory in JOBS.iterdir():
        state_file = directory / "state.json"
        if state_file.exists() and get_state(directory).get("project_id") == project_id:
            shutil.rmtree(directory)
    save_projects([p for p in projects if p["id"] != project_id])
    return {"deleted": project_id}


@app.get("/v1/jobs/{job_id}", dependencies=[Depends(auth)])
def job_status(job_id: str):
    return get_state(jdir(job_id))


@app.get("/v1/jobs/{job_id}/plan", dependencies=[Depends(auth)])
def get_plan(job_id: str):
    d = jdir(job_id)
    if not (d / "plan.json").exists():
        err(409, "PLAN_NOT_READY", "Plan is not ready yet", "Poll the job status")
    return json.loads((d / "plan.json").read_text())


@app.put("/v1/jobs/{job_id}/plan", dependencies=[Depends(auth)])
def put_plan(job_id: str, plan: dict):
    d = jdir(job_id)
    try:
        p = validate_plan(plan)
    except PlanError as e:
        err(422, "PLAN_INVALID", str(e), "Use only allow-listed types and transforms")
    (d / "plan.json").write_text(json.dumps(p.to_dict(), indent=2))
    set_state(d, plan_hash=p.hash())
    return {"plan_hash": p.hash()}


@app.post("/v1/jobs/{job_id}/run", dependencies=[Depends(auth)])
def run(job_id: str, bg: BackgroundTasks):
    d = jdir(job_id)
    if not (d / "plan.json").exists():
        err(409, "PLAN_NOT_READY", "No plan to run yet")
    set_state(d, status="transforming")
    bg.add_task(_run_job, d)
    return {"status": "transforming"}


@app.get("/v1/jobs/{job_id}/output", dependencies=[Depends(auth)])
def output(job_id: str, format: str = "csv"):
    d = jdir(job_id)
    if format not in ("csv", "json", "parquet"):
        err(400, "BAD_FORMAT", "format must be csv, json or parquet")
    f = d / f"output.{format}"
    if not f.exists():
        err(409, "NOT_READY", "Output not available yet", "Run the job first")
    return FileResponse(f, filename=f"output.{format}")


@app.get("/v1/jobs/{job_id}/report", dependencies=[Depends(auth)])
def report(job_id: str):
    d = jdir(job_id)
    if not (d / "report.json").exists():
        err(409, "NOT_READY", "Report not available yet")
    return json.loads((d / "report.json").read_text())


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/readyz")
def readyz():
    return {"ok": JOBS.exists(), "llm_configured": bool(settings.groq_api_key)}


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return (Path(__file__).parent.parent / "static" / "index.html").read_text()
