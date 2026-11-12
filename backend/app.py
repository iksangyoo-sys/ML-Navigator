"""Flask 서버: REST API + 빌드된 React 프런트엔드 제공."""
from __future__ import annotations

import hmac
import io
import os
import secrets
import time
import uuid
from functools import wraps
from pathlib import Path

from flask import Flask, Response, abort, jsonify, request, send_file, send_from_directory, session

from . import actions, ai, jobs, report
from .analysis import dataio, modeling, predict as PR
from .storage import DATA_DIR, HOME, Project, safe_name, to_jsonable
from .workspace import STEP_TITLES, WS

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "frontend" / "dist"
ALLOWED = {".csv", ".xlsx", ".xlsm", ".xls", ".parquet", ".tsv", ".txt"}


def secret_key() -> str:
    env = os.environ.get("SECRET_KEY")
    if env:
        return env
    p = HOME / "secret.key"
    if not p.exists():
        p.write_text(secrets.token_hex(32))
    return p.read_text().strip()


app = Flask(__name__, static_folder=None)
app.config.update(SECRET_KEY=secret_key(), MAX_CONTENT_LENGTH=1100 * 1024 * 1024, SESSION_COOKIE_SAMESITE="Lax",
                  SESSION_COOKIE_HTTPONLY=True, PERMANENT_SESSION_LIFETIME=60 * 60 * 24 * 7,
                  SESSION_COOKIE_SECURE=bool(os.environ.get("RENDER")))
PASSWORD = os.environ.get("APP_PASSWORD") or os.environ.get("ANALYSIS_PASSWORD")


def ok(data=None, **kw):
    return jsonify(to_jsonable({"ok": True, **({"data": data} if data is not None else {}), **kw}))


def err(msg, code=400):
    return jsonify({"ok": False, "error": msg}), code


def authed(fn):
    @wraps(fn)
    def wrap(*a, **kw):
        if PASSWORD and not session.get("auth"):
            return err("로그인이 필요합니다.", 401)
        return fn(*a, **kw)
    return wrap


def ws_or_404(pid) -> WS:
    try:
        return WS(pid)
    except KeyError:
        abort(404)


@app.errorhandler(ValueError)
def value_error(e):
    return err(str(e), 400)


@app.errorhandler(404)
def nf(e):
    if request.path.startswith("/api/"):
        return err("찾을 수 없습니다.", 404)
    return spa("")


@app.errorhandler(413)
def too_big(e):
    return err("파일이 너무 큽니다(입력 합계 1GB 이하).", 413)


# ------------------------------------------------------------- 인증
@app.get("/api/auth")
def auth_state():
    return ok({"required": bool(PASSWORD), "authed": (not PASSWORD) or bool(session.get("auth")), "ai": ai.available(),
               "ephemeral": bool(os.environ.get("EPHEMERAL_STORAGE"))})


@app.post("/api/login")
def login():
    if not PASSWORD:
        return ok({"authed": True})
    pw = (request.get_json(silent=True) or {}).get("password", "")
    if hmac.compare_digest(str(pw), PASSWORD):
        session.permanent = True
        session["auth"] = True
        return ok({"authed": True})
    time.sleep(0.8)
    return err("비밀번호가 올바르지 않습니다.", 401)


@app.post("/api/logout")
def logout():
    session.clear()
    return ok()


@app.get("/health")
def health():
    return {"status": "ok"}


# ------------------------------------------------------------- 프로젝트
@app.get("/api/projects")
@authed
def projects():
    return ok(Project.list())


@app.post("/api/projects")
@authed
def create_project():
    name = (request.get_json(silent=True) or {}).get("name") or "새 분석 프로젝트"
    p = Project.create(name[:80])
    return ok({"id": p.pid})


@app.patch("/api/projects/<pid>")
@authed
def rename_project(pid):
    ws = ws_or_404(pid)
    body = request.get_json(silent=True) or {}
    if body.get("name"):
        ws.p.rename(body["name"][:80])
    return ok(ws.summary())


@app.delete("/api/projects/<pid>")
@authed
def delete_project(pid):
    ws_or_404(pid).p.delete()
    return ok()


@app.get("/api/projects/<pid>")
@authed
def project(pid):
    return ok(ws_or_404(pid).summary())


@app.get("/api/projects/<pid>/results/<key>")
@authed
def result(pid, key):
    r = ws_or_404(pid).p.load_result(safe_name(key))
    return ok(r)


@app.get("/api/projects/<pid>/results")
@authed
def results_for_step(pid):
    ws = ws_or_404(pid)
    step = request.args.get("step")
    out = {}
    for k in ws.st["results_meta"]:
        if step is None or k == f"s{step}" or k.startswith(f"s{step}_"):
            out[k] = ws.p.load_result(k)
    return ok(out)


@app.put("/api/projects/<pid>/roles")
@authed
def set_roles(pid):
    ws = ws_or_404(pid)
    roles = ws.st["roles"]
    body = request.get_json(silent=True) or {}
    for c, r in (body.get("roles") or {}).items():
        if c in roles and r in ("feature", "id", "date", "group", "exclude") and roles[c] != "target":
            roles[c] = r
    ws.update(roles=roles)
    ws.mark_change(1, "열 역할 변경")
    return ok(ws.summary())


# ------------------------------------------------------------- 파일
@app.get("/api/data-files")
@authed
def data_files():
    out = []
    for f in sorted(DATA_DIR.iterdir()):
        if f.is_file() and f.suffix.lower() in ALLOWED:
            out.append({"name": f.name, "size": f.stat().st_size, "mtime": f.stat().st_mtime})
    return ok(out)


def add_source(ws: WS, path: Path, name: str, origin: str, encoding=None, sheet=None):
    total = sum(s.get("size", 0) for s in ws.st["sources"]) + path.stat().st_size
    if total > dataio.LIMIT_BYTES:
        raise ValueError("입력 파일 합계가 1GB를 넘습니다.")
    df, info = dataio.read_file(path, encoding, sheet)
    df = dataio.coerce_types(df)
    sid = uuid.uuid4().hex[:8]
    ws.p.save_df(f"src_{sid}", df)
    src = {"id": sid, "name": name, "origin": origin, "rows": len(df), "cols": df.shape[1], "size": path.stat().st_size,
           "hash": dataio.file_hash(path), "added": time.time(), **{k: v for k, v in info.items() if k != "sheets"},
           "sheets": info.get("sheets")}
    ws.update(sources=ws.st["sources"] + [src])
    if len(ws.st["sources"]) == 1:
        ws.set_merged(df, {"mode": "single", "sources": [sid]}, step=1)
    else:
        ws.mark_change(1, f"파일 추가: {name}")
    return src


@app.post("/api/projects/<pid>/upload")
@authed
def upload(pid):
    ws = ws_or_404(pid)
    files = request.files.getlist("files")
    if not files:
        return err("파일을 선택하세요.")
    added = []
    updir = ws.p.dir / "sources"
    for f in files:
        name = safe_name(f.filename or "upload.csv")
        if Path(name).suffix.lower() not in ALLOWED:
            return err(f"{name}: CSV, XLSX, Parquet만 지원합니다.")
        dest = updir / f"{uuid.uuid4().hex[:6]}_{name}"
        f.save(dest)
        try:
            added.append(add_source(ws, dest, name, "upload", request.form.get("encoding") or None, request.form.get("sheet") or None))
        except Exception as e:
            dest.unlink(missing_ok=True)
            return err(f"{name}: {e}")
    return ok(ws.summary(), added=added)


@app.post("/api/projects/<pid>/sources/server")
@authed
def add_server_file(pid):
    ws = ws_or_404(pid)
    body = request.get_json(silent=True) or {}
    name = Path(body.get("name", "")).name
    path = DATA_DIR / name
    if not path.is_file() or path.suffix.lower() not in ALLOWED:
        return err("data 폴더에서 파일을 찾을 수 없습니다.")
    src = add_source(ws, path, name, "server", body.get("encoding") or None, body.get("sheet") or None)
    return ok(ws.summary(), added=[src])


@app.post("/api/projects/<pid>/sources/<sid>/reload")
@authed
def reload_source(pid, sid):
    """인코딩·시트를 바꿔 다시 읽기."""
    ws = ws_or_404(pid)
    body = request.get_json(silent=True) or {}
    srcs = ws.st["sources"]
    s = next((x for x in srcs if x["id"] == sid), None)
    if not s:
        return err("원본을 찾을 수 없습니다.")
    path = DATA_DIR / s["name"] if s["origin"] == "server" else next((ws.p.dir / "sources").glob(f"*_{s['name']}"), None)
    if not path or not path.exists():
        return err("원본 파일이 없습니다.")
    df, info = dataio.read_file(path, body.get("encoding") or None, body.get("sheet") or None)
    df = dataio.coerce_types(df)
    ws.p.save_df(f"src_{sid}", df)
    s.update({"rows": len(df), "cols": df.shape[1], **{k: v for k, v in info.items() if k != "sheets"}})
    ws.update(sources=srcs)
    if (ws.st.get("merge") or {}).get("mode") == "single" and (ws.st["merge"].get("sources") or [None])[0] == sid:
        ws.set_merged(df, ws.st["merge"], step=1)
    ws.mark_change(1, f"다시 읽기: {s['name']}")
    return ok(ws.summary())


@app.delete("/api/projects/<pid>/sources/<sid>")
@authed
def delete_source(pid, sid):
    ws = ws_or_404(pid)
    srcs = [s for s in ws.st["sources"] if s["id"] != sid]
    ws.update(sources=srcs)
    ws.p.drop_df(f"src_{sid}")
    if srcs and (ws.st.get("merge") or {}).get("sources") and sid in ws.st["merge"]["sources"]:
        ws.set_merged(ws.source_df(srcs[0]["id"]), {"mode": "single", "sources": [srcs[0]["id"]]}, step=1)
    ws.mark_change(1, "파일 삭제")
    return ok(ws.summary())


@app.post("/api/projects/<pid>/newdata")
@authed
def upload_newdata(pid):
    ws = ws_or_404(pid)
    f = request.files.get("file")
    if not f:
        return err("파일을 선택하세요.")
    name = safe_name(f.filename or "new.csv")
    dest = ws.p.dir / "sources" / f"new_{uuid.uuid4().hex[:6]}_{name}"
    f.save(dest)
    df, info = dataio.read_file(dest, request.form.get("encoding") or None)
    df = dataio.coerce_types(df)
    nid = uuid.uuid4().hex[:8]
    ws.p.save_df(f"new_{nid}", df)
    nd = {"id": nid, "name": name, "rows": len(df), "cols": df.shape[1], "encoding": info.get("encoding"), "added": time.time()}
    ws.update(newdata=(ws.st.get("newdata") or []) + [nd])
    return ok(ws.summary(), added=nd)


# ------------------------------------------------------------- 메타 정보
@app.get("/api/projects/<pid>/models")
@authed
def model_catalog(pid):
    ws = ws_or_404(pid)
    tt = ws.st.get("target_type") or "regression"
    return ok({"models": modeling.model_list(tt), "select_metrics": list(modeling.SELECT_METRICS[tt].keys())})


@app.get("/api/projects/<pid>/input-spec")
@authed
def input_spec(pid):
    ws = ws_or_404(pid)
    ctx = actions.model_ctx(ws, request.args.get("model"))
    return ok({"spec": ctx["spec"], "model": ctx["model_name"], "classes": ctx["classes"], "ttype": ctx["ttype"]})


@app.get("/api/projects/<pid>/sample-row")
@authed
def sample_row(pid):
    """학습 데이터에서 행 하나(무작위 또는 목표 미달 행)를 입력값으로 가져온다."""
    ws = ws_or_404(pid)
    ctx = actions.model_ctx(ws)
    bt = ctx["base_train"]
    i = int(request.args.get("index", -1))
    row = bt.iloc[i] if 0 <= i < len(bt) else bt.sample(1, random_state=int(time.time()) % 10000).iloc[0]
    y = ctx["y_base_train"].loc[row.name]
    def clean(v):
        if isinstance(v, float):
            return None if v != v else float(f"{v:.6g}")
        return v
    return ok({"row": {k: clean(v) for k, v in row.to_dict().items()}, "actual": y, "index": int(row.name)})


# ------------------------------------------------------------- 작업 실행
@app.post("/api/projects/<pid>/run/<name>")
@authed
def run(pid, name):
    ws_or_404(pid)
    if name not in actions.REGISTRY:
        return err(f"알 수 없는 작업: {name}")
    params = request.get_json(silent=True) or {}
    job = jobs.submit(pid, name, params, actions.run_action)
    return ok({"job": job.id})


@app.get("/api/jobs/<jid>")
@authed
def job_status(jid):
    j = jobs.get(jid)
    if not j:
        return err("작업을 찾을 수 없습니다.", 404)
    return ok(j.as_dict())


@app.post("/api/jobs/<jid>/cancel")
@authed
def job_cancel(jid):
    j = jobs.get(jid)
    if not j:
        return err("작업을 찾을 수 없습니다.", 404)
    j.cancel()
    return ok(j.as_dict(False))


@app.get("/api/projects/<pid>/jobs")
@authed
def project_jobs(pid):
    return ok(jobs.for_project(pid))


# ------------------------------------------------------------- 내려받기·보고서
@app.get("/api/projects/<pid>/download/<name>")
@authed
def download(pid, name):
    ws = ws_or_404(pid)
    name = safe_name(name)
    if name in ("working", "merged", "view"):
        df = ws.view() if name == "view" else (ws.working() if name == "working" else ws.merged())
    elif name.startswith("pred_") and ws.p.has_df(name):
        df = ws.p.load_df(name)
    elif name.startswith("model_"):
        key = name[6:]
        if not ws.p.has_obj(f"pipe_{key}"):
            abort(404)
        return send_file(ws.p.dir / "models" / f"pipe_{key}.joblib", as_attachment=True, download_name=f"{key}_pipeline.joblib")
    else:
        abort(404)
    buf = io.BytesIO(df.to_csv(index=False).encode("utf-8-sig"))
    return send_file(buf, mimetype="text/csv", as_attachment=True, download_name=f"{name}.csv")


@app.get("/api/projects/<pid>/report/steps")
@authed
def report_steps(pid):
    ws = ws_or_404(pid)
    keys = [k for k in request.args.get("keys", "").split(",") if k] or None
    html_ = report.step_report(ws, keys)
    ws.p.dir.joinpath("reports", f"steps_{int(time.time())}.html").write_text(html_, encoding="utf-8")
    return Response(html_, mimetype="text/html")


@app.post("/api/projects/<pid>/report/executive")
@authed
def report_exec(pid):
    ws = ws_or_404(pid)
    fields = request.get_json(silent=True) or {}
    ws.update(exec_fields=fields)
    html_ = report.executive(ws, fields)
    ws.p.dir.joinpath("reports", f"executive_{int(time.time())}.html").write_text(html_, encoding="utf-8")
    return ok({"html": html_})


@app.get("/api/projects/<pid>/report/executive-fields")
@authed
def exec_fields(pid):
    ws = ws_or_404(pid)
    st = ws.st
    f = st.get("exec_fields") or {}
    if not f:
        best = next((m for m in st.get("models") or [] if m["key"] == st.get("best_model")), None)
        base = next((m for m in st.get("models") or [] if m["key"] == "baseline"), None)
        kpi = ""
        if best and st.get("target_type") == "regression" and base:
            kpi = f"예측 오차(RMSE) 기준선 {base['test']['RMSE']:.4g} → {best['name']} {best['test']['RMSE']:.4g}, R² {best['test']['R2_Score']:.3f}"
        elif best and base:
            kpi = f"정확도 기준선 {base['test']['Accuracy']:.3f} → {best['name']} {best['test']['Accuracy']:.3f}" + (f", AUC {best['test'].get('AUC'):.3f}" if best['test'].get('AUC') else "")
        f = {"question": f"{st.get('target') or '종속변수'}을(를) 개선하려면 어떤 조건을 조정해야 하는가?", "kpi": kpi,
             "uncertainty": "관측 데이터 기반 분석으로, 측정되지 않은 요인이 결과에 영향을 줄 수 있습니다. 권고 조건은 확인실험으로 검증이 필요합니다."}
    return ok(f)


@app.post("/api/projects/<pid>/ai/explain")
@authed
def ai_explain(pid):
    ws = ws_or_404(pid)
    body = request.get_json(silent=True) or {}
    key = safe_name(body.get("key", ""))
    r = ws.p.load_result(key)
    if not r:
        return err("해설할 결과가 없습니다.")
    from .workspace import step_of
    st = ws.st
    text = ai.explain(STEP_TITLES.get(step_of(key), key), r["blocks"],
                      {"target": st.get("target"), "target_type": st.get("target_type"), "shape": st.get("working_shape")})
    return ok({"text": text})


# ------------------------------------------------------------- 프런트엔드
@app.get("/")
@app.get("/<path:path>")
def spa(path=""):
    if path.startswith("api/"):
        abort(404)
    if DIST.exists():
        target = DIST / path
        if path and target.is_file():
            return send_from_directory(DIST, path)
        return send_from_directory(DIST, "index.html")
    return Response("<h3>프런트엔드가 빌드되지 않았습니다. frontend 폴더에서 npm install && npm run build를 실행하세요.</h3>", mimetype="text/html")


def main():
    port = int(os.environ.get("PORT", 5000))
    print(f"\n  빅데이터 분석 자동화 웹: http://127.0.0.1:{port}\n")
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
