#! python3.12
"""Flask interface and durable run storage."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hmac
from importlib.machinery import EXTENSION_SUFFIXES
import json
import logging
import math
import os
from pathlib import Path
import re
import sqlite3
import sys
from uuid import uuid4

VENDOR = Path(__file__).resolve().parent / ".vendor"
if VENDOR.is_dir() and sys.prefix == sys.base_prefix:
    numpy_core = VENDOR / "numpy" / "_core"
    if any((numpy_core / f"_multiarray_umath{suffix}").is_file() for suffix in EXTENSION_SUFFIXES):
        sys.path.insert(0, str(VENDOR))
    elif __name__ == "__main__" and os.name == "nt":
        raise SystemExit(
            f"현재 Python은 {sys.version.split()[0]}입니다. "
            "이 작업 폴더의 로컬 패키지는 64비트 Python 3.12용입니다. "
            "Python 3.12에서 app.py를 실행하세요. 자세한 명령은 README.md를 참고하세요."
        )

from flask import Flask, abort, jsonify, redirect, render_template, request, send_from_directory, url_for
import numpy as np
import pandas as pd
import joblib
from werkzeug.utils import secure_filename

from analysis_engine import FORMATS, LIMIT, batch, explore, model, prepare, read
from ai_analysis import REPORT_FIELDS, recommend, report_draft
from method_catalog import METHODS
from quality_engine import design, doe_results, spc


BASE = Path(__file__).resolve().parent
ROOT = Path(os.getenv("ANALYSIS_HOME", BASE / "instance")).resolve()
ROOT.mkdir(parents=True, exist_ok=True)
PASSWORD = os.getenv("ANALYSIS_PASSWORD")
PUBLIC_ORIGIN = os.getenv("PUBLIC_ORIGIN") or os.getenv("RENDER_EXTERNAL_URL", "")
PROJECT_PASSWORDS = json.loads(os.getenv("ANALYSIS_PROJECT_PASSWORDS", "{}"))
if not isinstance(PROJECT_PASSWORDS, dict) or any(not isinstance(key, str) or not isinstance(value, str) or not value for key, value in PROJECT_PASSWORDS.items()):
    raise RuntimeError("ANALYSIS_PROJECT_PASSWORDS는 프로젝트별 비밀번호의 JSON 객체입니다.")
if os.getenv("RENDER") and not (PASSWORD or PROJECT_PASSWORDS):
    raise RuntimeError("Render에서는 접근 비밀번호가 필요합니다.")
app = Flask(__name__, instance_path=str(ROOT))
app.config["MAX_CONTENT_LENGTH"] = LIMIT + 1024 * 1024
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="analysis")
logger = logging.getLogger(__name__)
KINDS = {"eda": "전처리·EDA", "model": "모델링·검증", "batch": "배치 모니터링",
         "spc": "공정관리", "design": "DOE 설계", "doe": "DOE 결과·확인실험"}


def database(query, parameters=(), fetch=False):
    with sqlite3.connect(ROOT / "runs.db", timeout=20) as connection:
        connection.row_factory = sqlite3.Row
        cursor = connection.execute(query, parameters)
        return [dict(row) for row in cursor.fetchall()] if fetch else cursor.rowcount


database("""CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY, project TEXT NOT NULL, kind TEXT NOT NULL, state TEXT NOT NULL,
    created TEXT NOT NULL, config TEXT NOT NULL, sources TEXT NOT NULL,
    error TEXT, approved TEXT, narrative TEXT
)""")
database("UPDATE runs SET state='interrupted' WHERE state IN ('queued','running')")
database("""CREATE TABLE IF NOT EXISTS reports (
    id TEXT PRIMARY KEY, project TEXT NOT NULL, created TEXT NOT NULL, records TEXT NOT NULL
)""")


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return clean(value.item())
    if value is pd.NA or value is pd.NaT or (isinstance(value, float) and not math.isfinite(value)):
        return None
    return value


def write_json(path, data):
    path.write_text(json.dumps(clean(data), ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")


def project_name(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,59}", value):
        abort(400, "프로젝트는 영문·숫자·하이픈·밑줄 1~60자입니다.")
    return value


def run_path(run_id):
    return ROOT / "runs" / run_id


def find_run(project, run_id):
    matches = database("SELECT * FROM runs WHERE id=? AND project=?", (run_id, project), True)
    if not matches:
        abort(404)
    return matches[0]


def input_file(project, source):
    prefix, _, name = source.partition(":")
    if Path(name).name != name or not name or Path(name).suffix.lower() not in FORMATS:
        raise ValueError("파일명이 올바르지 않습니다.")
    if prefix not in {"data", "upload"}:
        raise ValueError("파일 위치가 올바르지 않습니다.")
    folder = BASE / "data" if prefix == "data" else ROOT / "projects" / project
    path = folder / name
    if not path.is_file() or path.resolve().parent != folder.resolve():
        raise ValueError(f"파일을 찾을 수 없습니다: {name}")
    return path


def run_analysis(project, run_id, kind, config, sources):
    changed = database("UPDATE runs SET state='running' WHERE id=? AND state='queued'", (run_id,))
    if not changed:
        return
    directory = run_path(run_id)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        dataset_cfg = config.get("data", {})
        settings = config.get("options", {})
        if kind == "batch":
            parent = find_run(project, settings["model_run"])
            if parent["kind"] != "model" or parent["state"] != "complete" or not parent["approved"]:
                raise ValueError("승인한 모델 실행을 선택하세요.")
            if not dataset_cfg:
                parent_config = json.loads(parent["config"])
                dataset_cfg = parent_config.get("data", {})
                dataset_cfg.pop("base_source", None)
                dataset_cfg = {**dataset_cfg, "rules": [
                    rule for rule in dataset_cfg.get("rules", [])
                    if rule["column"] != parent_config["options"]["target"]
                ]}
        result = {}
        if kind == "design":
            result["design"], plan = design(settings)
            plan.to_csv(directory / "design.csv", index=False, encoding="utf-8-sig")
        else:
            if dataset_cfg.get("base_source"):
                base_source = dataset_cfg["base_source"]
                if base_source not in sources:
                    raise ValueError("기준 파일은 선택한 원본에 포함되어야 합니다.")
                sources = [base_source, *[item for item in sources if item != base_source]]
            paths = [input_file(project, item) for item in sources]
            frame, versions, audit = prepare(paths, dataset_cfg)
            result.update({"input": versions, "preprocessing": audit, "rows": len(frame), "columns": list(frame.columns)})
            if kind == "eda":
                result["eda"] = explore(frame, settings)
            elif kind == "model":
                result["model"], predictions, artifact, development = model(frame, settings)
                result["eda"] = explore(development, config.get("eda", {}))
                predictions.to_csv(directory / "predictions.csv", index=False, encoding="utf-8-sig")
                joblib.dump(artifact, directory / "model.joblib")
            elif kind == "batch":
                artifact = joblib.load(run_path(parent["id"]) / "model.joblib")
                result["batch"], predictions = batch(frame, artifact, settings)
                predictions.to_csv(directory / "predictions.csv", index=False, encoding="utf-8-sig")
            elif kind == "spc":
                result["spc"], table = spc(frame, settings)
                table.to_csv(directory / "spc.csv", index=False, encoding="utf-8-sig")
            elif kind == "doe":
                parent = find_run(project, settings["design_run"])
                if parent["kind"] != "design" or parent["state"] != "complete" or not parent["approved"]:
                    raise ValueError("승인한 실험설계를 선택하세요.")
                plan = pd.read_csv(run_path(parent["id"]) / "design.csv")
                result["doe"], table = doe_results(plan, frame, json.loads(parent["config"])["options"], settings)
                table.to_csv(directory / "observations.csv", index=False, encoding="utf-8-sig")
        write_json(directory / "result.json", result)
        write_json(directory / "config.json", config)
        database("UPDATE runs SET state='complete' WHERE id=? AND state='running'", (run_id,))
    except Exception as error:
        logger.exception("Run %s failed", run_id)
        message = error.description if hasattr(error, "description") else str(error)
        database("UPDATE runs SET state='failed', error=? WHERE id=? AND state='running'", (message, run_id))


@app.before_request
def authorize():
    if request.endpoint in {"health", "static"}:
        return None
    project = (request.view_args or {}).get("project")
    required = PROJECT_PASSWORDS.get(project) if PROJECT_PASSWORDS and project else PASSWORD
    if PROJECT_PASSWORDS and project and not required:
        abort(404)
    if required:
        auth = request.authorization
        if not auth or auth.type != "basic" or not hmac.compare_digest((auth.password or "").encode(), required.encode()):
            return "인증이 필요합니다.", 401, {"WWW-Authenticate": 'Basic realm="Analysis"'}
    if request.method == "POST":
        if request.headers.get("Sec-Fetch-Site") == "cross-site":
            abort(403)
        origin = request.headers.get("Origin")
        if origin and origin.rstrip("/") != request.host_url.rstrip("/") and origin.rstrip("/") != PUBLIC_ORIGIN.rstrip("/"):
            abort(403)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def index():
    return redirect(url_for("workspace", project=next(iter(PROJECT_PASSWORDS), "default")))


@app.get("/p/<project>")
def workspace(project):
    project_name(project)
    sources = []
    for prefix, folder in [("data", BASE / "data"), ("upload", ROOT / "projects" / project)]:
        if folder.is_dir():
            sources.extend(f"{prefix}:{path.name}" for path in sorted(folder.iterdir())
                           if path.is_file() and path.suffix.lower() in FORMATS)
    runs = database("SELECT id,kind,state,created,approved FROM runs WHERE project=? ORDER BY created DESC LIMIT 100",
                    (project,), True)
    return render_template("index.html", project=project, sources=sources, runs=runs, kinds=KINDS, methods=METHODS)


@app.post("/p/<project>/advice")
def advice(project):
    project_name(project)
    body = request.get_json()
    try:
        kind = body["kind"]
        sources = body.get("sources", [])
        if kind not in KINDS or not isinstance(sources, list) or len(sources) > 10 or (not sources and kind != "design"):
            raise ValueError("분석 단계와 원본 파일을 확인하세요. 추천에는 원본을 최대 10개 선택할 수 있습니다.")
        profiles = []
        read_options = body.get("data", {}).get("read", {})
        for index, source in enumerate(sources, 1):
            path = input_file(project, source)
            sample = read(path, read_options.get(path.name), preview_rows=2000)
            columns = []
            for name, values in sample.iloc[:, :100].items():
                info = {"name": str(name), "type": str(values.dtype),
                        "valid": int(values.count()), "distinct": int(values.nunique())}
                if pd.api.types.is_numeric_dtype(values):
                    finite = values[np.isfinite(values)]
                    info["skew"] = finite.skew() if len(finite) >= 3 else None
                elif 1 < info["distinct"] <= 30:
                    info["group_counts"] = values.value_counts().tolist()
                columns.append(info)
            profiles.append({"source_index": index, "sample_rows": len(sample), "columns": columns})
        return jsonify(recommend(kind, clean(profiles), str(body.get("objective", ""))[:2500]))
    except (ValueError, KeyError, TypeError, AttributeError, OSError) as error:
        return jsonify(error=str(error)), 400


@app.post("/p/<project>/runs/<run_id>/draft")
def draft_run(project, run_id):
    run = find_run(project, run_id)
    if run["state"] != "complete" or run["approved"]:
        abort(409, "승인 전 완료 실행에서 보고서 초안을 작성할 수 있습니다.")
    result = json.loads((run_path(run_id) / "result.json").read_text(encoding="utf-8"))
    evidence = {"stage": KINDS[run["kind"]], "rows": result.get("rows"),
                "preprocessing": result.get("preprocessing", [])[:20]}
    for section, keys in {
        "eda": ("rows", "chart_rows", "test"),
        "model": ("selected", "cv_scoring", "train_rows", "test_rows", "candidates", "test"),
        "batch": ("rows", "label_coverage", "metrics"),
        "spc": ("chart", "center", "reference", "signals", "capability"),
        "design": ("design", "planned", "excluded", "rank", "terms", "residual_df", "resolution"),
        "doe": ("planned", "observed", "anova", "confirmation"),
    }.items():
        if section in result:
            evidence[section] = {key: result[section][key] for key in keys if key in result[section]}
    if "eda" in result:
        evidence["eda"]["numeric_summary"] = [
            {key: value for key, value in column.items() if key != "top"} for column in result["eda"]["summary"][:20]
        ]
    evidence["objective"] = json.loads(run["config"]).get("task_definition", {}).get("question", "")[:2500]
    try:
        return jsonify(report_draft(evidence))
    except ValueError as error:
        return jsonify(error=str(error)), 400


@app.post("/p/<project>/upload")
def upload(project):
    project_name(project)
    file = request.files.get("file")
    if not file or Path(file.filename).suffix.lower() not in FORMATS:
        abort(400, "CSV, XLSX 또는 Parquet를 선택하세요.")
    folder = ROOT / "projects" / project
    folder.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename).suffix.lower()
    stem = secure_filename(Path(file.filename).stem) or "dataset"
    name = f"{uuid4().hex[:12]}_{stem[:70]}{suffix}"
    temporary = folder / (name + ".part")
    try:
        total = 0
        with temporary.open("xb") as output:
            for chunk in iter(lambda: file.stream.read(1024 * 1024), b""):
                total += len(chunk)
                if total > LIMIT:
                    abort(413)
                output.write(chunk)
        if not total:
            abort(400, "빈 파일입니다.")
        temporary.replace(folder / name)
    finally:
        temporary.unlink(missing_ok=True)
    return redirect(url_for("workspace", project=project), code=303)


@app.post("/p/<project>/runs")
def launch(project):
    project_name(project)
    kind = request.form.get("kind")
    if kind not in KINDS:
        abort(400, "작업을 선택하세요.")
    try:
        config = json.loads(request.form.get("config", "{}"))
        if not isinstance(config, dict):
            raise ValueError("설정은 JSON 객체여야 합니다.")
    except ValueError as error:
        abort(400, f"설정 오류: {error}")
    sources = request.form.getlist("source")
    if kind != "design" and not sources:
        abort(400, "데이터 파일을 선택하세요.")
    if kind == "design":
        sources = []
    run_id = uuid4().hex
    database("INSERT INTO runs (id,project,kind,state,created,config,sources) VALUES (?,?,?,?,?,?,?)",
             (run_id, project, kind, "queued", datetime.now(timezone.utc).isoformat(), json.dumps(config, ensure_ascii=False), json.dumps(sources)))
    executor.submit(run_analysis, project, run_id, kind, config, sources)
    return redirect(url_for("detail", project=project, run_id=run_id), code=303)


@app.get("/p/<project>/runs/<run_id>")
def detail(project, run_id):
    run = find_run(project, run_id)
    directory = run_path(run_id)
    result = json.loads((directory / "result.json").read_text(encoding="utf-8")) if run["state"] == "complete" else None
    charts = []
    sections = []
    if result:
        for key, value in result.items():
            if isinstance(value, dict):
                value = value.copy()
                chart = value.pop("chart_data", None) or (value.pop("chart", None) if key == "eda" else None)
                if chart:
                    charts.append(chart)
            sections.append((key, json.dumps(value, ensure_ascii=False, indent=2)))
    files = [name for name in DOWNLOADS if (directory / name).is_file()] if result else []
    return render_template("run.html", run=run, project=project, result=result, sections=sections,
                           charts=charts, files=files, kind=KINDS[run["kind"]])


@app.post("/p/<project>/runs/<run_id>/action")
def action(project, run_id):
    run = find_run(project, run_id)
    operation = request.form.get("operation")
    if operation == "cancel":
        database("UPDATE runs SET state='cancelled' WHERE id=? AND state IN ('queued','running')", (run_id,))
    elif operation == "approve":
        narrative = {field: request.form.get(field, "").strip()[:4000] for field in
                     REPORT_FIELDS}
        database("UPDATE runs SET approved=?,narrative=? WHERE id=? AND state='complete' AND approved IS NULL",
                 (datetime.now(timezone.utc).isoformat(), json.dumps(narrative, ensure_ascii=False), run_id))
    elif operation == "retry":
        new_id = uuid4().hex
        database("INSERT INTO runs (id,project,kind,state,created,config,sources) VALUES (?,?,?,?,?,?,?)",
                 (new_id, project, run["kind"], "queued", datetime.now(timezone.utc).isoformat(), run["config"], run["sources"]))
        executor.submit(run_analysis, project, new_id, run["kind"], json.loads(run["config"]), json.loads(run["sources"]))
        return redirect(url_for("detail", project=project, run_id=new_id), code=303)
    else:
        abort(400)
    return redirect(url_for("detail", project=project, run_id=run_id), code=303)


DOWNLOADS = {"config.json", "result.json", "predictions.csv", "spc.csv", "design.csv", "observations.csv"}


@app.get("/p/<project>/runs/<run_id>/files/<name>")
def download(project, run_id, name):
    run = find_run(project, run_id)
    if run["state"] != "complete" or name not in DOWNLOADS:
        abort(404)
    return send_from_directory(run_path(run_id), name, as_attachment=True)


def approved_records(project):
    runs = database("SELECT * FROM runs WHERE project=? AND approved IS NOT NULL ORDER BY approved DESC", (project,), True)
    records = []
    for run in runs:
        summary = json.loads((run_path(run["id"]) / "result.json").read_text(encoding="utf-8"))
        metrics = {}
        if run["kind"] == "eda" and "test" in summary.get("eda", {}):
            test = summary["eda"]["test"]
            metrics = {"검정": test["method"], "p값": test.get("p_value", test.get("p")),
                       "효과 추정": test.get("effect")}
        elif "model" in summary:
            metrics = {"선택 모델": summary["model"]["selected"], **{k: v for k, v in summary["model"]["test"].items() if isinstance(v, (int, float))}}
        elif "spc" in summary:
            metrics = {"중심선": summary["spc"]["center"], "신호 건수": summary["spc"]["signals"]}
        elif "batch" in summary:
            metrics = {"배치 건수": summary["batch"]["rows"], "라벨 커버리지": summary["batch"]["label_coverage"]}
        elif "design" in summary:
            metrics = {"계획 실행 수": summary["design"]["planned"], "설계 행렬 계수": summary["design"]["rank"]}
        elif "doe" in summary:
            metrics = {"실험 결과 건수": summary["doe"]["observed"], "확인실험": summary["doe"].get("confirmation")}
        records.append((run, json.loads(run["narrative"]), metrics))
    return records


@app.get("/p/<project>/report")
def report(project):
    project_name(project)
    records = approved_records(project)
    snapshots = database("SELECT id,created FROM reports WHERE project=? ORDER BY created DESC", (project,), True)
    return render_template("report.html", project=project, records=records, kinds=KINDS, snapshots=snapshots)


@app.post("/p/<project>/report/draft")
def draft_project(project):
    project_name(project)
    records = approved_records(project)[:30]
    if not records:
        return jsonify(error="먼저 실행 결과를 승인하세요."), 400
    evidence = [{"stage": KINDS[run["kind"]], "metrics": metrics,
                 "approved_notes": {key: value[:1000] for key, value in narrative.items()}}
                for run, narrative, metrics in records]
    try:
        return jsonify(report_draft({"stage": "프로젝트 경영 요약", "approved_runs": evidence}))
    except ValueError as error:
        return jsonify(error=str(error)), 400


@app.post("/p/<project>/report")
def save_report(project):
    project_name(project)
    records = approved_records(project)
    if not records:
        abort(400, "보고서에 추가할 승인 결과가 없습니다.")
    executive = {field: request.form.get(field, "").strip()[:4000] for field in REPORT_FIELDS}
    identifier = uuid4().hex
    database("INSERT INTO reports VALUES (?,?,?,?)",
             (identifier, project, datetime.now(timezone.utc).isoformat(),
              json.dumps({"records": records, "executive": executive}, ensure_ascii=False)))
    return redirect(url_for("saved_report", project=project, identifier=identifier), code=303)


@app.get("/p/<project>/report/<identifier>")
def saved_report(project, identifier):
    project_name(project)
    reports = database("SELECT * FROM reports WHERE project=? AND id=?", (project, identifier), True)
    if not reports:
        abort(404)
    snapshot = reports[0]
    content = json.loads(snapshot["records"])
    if isinstance(content, list):
        content = {"records": content, "executive": {}}
    return render_template("report.html", project=project, kinds=KINDS, snapshot=snapshot, **content)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "5000")), threaded=True)
