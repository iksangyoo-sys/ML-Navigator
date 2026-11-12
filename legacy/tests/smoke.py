"""End-to-end Flask smoke check for each analysis path."""
import io
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["ANALYSIS_HOME"] = str(ROOT / "instance" / f"smoke-{uuid4().hex}")

import app
import numpy as np
import pandas as pd
from analysis_engine import prepare
from quality_engine import design, spc
from scipy.stats import ttest_ind

client = app.app.test_client()
project = "/p/verification"
assert client.get(project).status_code == 200


def upload(frame, name):
    folder = app.ROOT / "projects" / "verification"
    before = set(folder.glob("*"))
    payload = (io.BytesIO(frame.to_csv(index=False).encode()), name)
    response = client.post(project + "/upload", data={"file": payload})
    assert response.status_code == 303, response.data
    return "upload:" + (set(folder.glob("*")) - before).pop().name


def run(kind, config, sources=(), approve=False):
    response = client.post(project + "/runs", data={"kind": kind, "config": json.dumps(config), "source": list(sources)})
    assert response.status_code == 303, response.data
    route = response.location
    identifier = route.rsplit("/", 1)[-1]
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        record = app.database("SELECT * FROM runs WHERE id=?", (identifier,), True)[0]
        if record["state"] not in ("queued", "running"):
            break
        time.sleep(.1)
    assert record["state"] == "complete", record
    assert client.get(route).status_code == 200
    if approve:
        response = client.post(route + "/action", data={"operation": "approve", "decision": "verify"})
        assert response.status_code == 303
    result = client.get(route + "/files/result.json").json
    print(f"{kind}: {identifier}", flush=True)
    return identifier, result


rng = np.random.default_rng(42)
size = 140
x = rng.normal(size=size)
frame = pd.DataFrame({"x": x, "category": np.where(x > 0, "A", "B"), "y": x*3 + rng.normal(0, .1, size),
                      "time": pd.date_range("2026-01-01", periods=size), "lot": np.arange(size)//7})
frame.loc[5, "category"] = None
src = upload(frame, "sample.csv")
with patch("ai_analysis.ask", return_value={"recommendations": [
    {"method": "eda:welch", "reason": "두 독립 그룹", "assumptions": "독립성",
     "assignments": [{"role": "group", "column": "category"}, {"role": "y", "column": "y"}]}
], "notes": ""}):
    advice = client.post(project + "/advice", json={"kind": "eda", "sources": [src], "objective": "group means"})
assert advice.status_code == 200
choice = advice.json["recommendations"][0]
options = {**choice["patch"], **{item["role"]: item["column"] for item in choice["assignments"]}}
eda_id, eda = run("eda", {"options": options}, [src])
group_a, group_b = [part["y"] for _, part in frame.dropna(subset=["category"]).groupby("category")]
assert np.isclose(eda["eda"]["test"]["p_value"], ttest_ind(group_a, group_b, equal_var=False).pvalue)
with patch("app.report_draft", return_value={"kpi": "Welch p-value"}) as draft:
    response = client.post(f"{project}/runs/{eda_id}/draft")
    assert response.json["kpi"] == "Welch p-value"
    assert draft.call_args.args[0]["eda"]["test"]["method"] == "welch"
client.post(f"{project}/runs/{eda_id}/action", data={"operation": "approve", "kpi": "Welch"})
reg, scores = run("model", {"options": {"target": "y", "features": ["x", "category"], "models": ["linear"],
                                      "folds": 3, "split": "group", "group": "lot"}}, [src], True)
assert scores["model"]["test"]["r2"] > .95, scores["model"]
classified = frame.assign(label=np.where(frame["x"] > 0, "good", "bad"))
_, classification = run("model", {"options": {"task": "classification", "target": "label",
    "features": ["x", "category"], "models": ["linear"], "split": "stratified", "folds": 3}},
    [upload(classified, "classification.csv")])
assert classification["model"]["test"]["f1_macro"] > .9
_, drift = run("batch", {"options": {"model_run": reg, "drift": "psi"}}, [src], True)
assert drift["batch"]["features"][0]["psi"] >= 0
run("spc", {"options": {"value": "y", "time": "time", "start": "2026-01-01",
                        "end": "2026-03-01", "lsl": -10, "usl": 10}}, [src], True)
plan_id, _ = run("design", {"options": {"design": "ccd", "repeat": 2, "center": 3,
                                       "factors": [{"name": "A", "low": 1, "high": 3},
                                                   {"name": "B", "low": 2, "high": 4}]}}, approve=True)
plan = pd.read_csv(app.run_path(plan_id) / "design.csv")
obs = pd.DataFrame({"run_id": plan["run_id"], "response": 3 + 2*plan["coded:A"] + plan["coded:B"]**2 + rng.normal(0, .01, len(plan))})
run("doe", {"options": {"design_run": plan_id, "response": "response",
                       "confirmation": {"baseline": [1, 1.1, 1.2], "candidate": [3.0, 3.1, 3.2], "minimum": 1}}},
    [upload(obs, "observations.csv")], True)
fractional, table = design({"design": "fractional", "generators": {"C": ["A", "B"]},
    "factors": [{"name": name, "low": -1, "high": 1} for name in "ABC"]})
assert fractional["resolution"] == 3 and fractional["main_effect_aliases"]["A"] == [["B", "C"]]
assert np.all(table["C"] == table["A"] * table["B"])
counts = pd.DataFrame({"time": ["2026-01-01", "2026-01-02", "2026-01-03"],
                       "lot": [1, 2, 3], "defective": [1, 4, 1], "total": [10, 20, 50]})
reference, limits = spc(counts, {"chart": "p", "value": "defective", "time": "time",
    "group": "lot", "denominator": "total", "start": "2026-01-01", "end": "2026-01-02"})
assert np.isclose(reference["center"], 5/30) and limits["ucl"].iloc[2] < limits["ucl"].iloc[0]
assert limits["phase"].tolist() == ["I", "I", "II"]
left = upload(pd.DataFrame({"key": [1, 2, None], "x": [1, 2, 3]}), "left.csv")
right = upload(pd.DataFrame({"key": [1, 3, None], "y": [4, 5, 6]}), "right.csv")
paths = [app.input_file("verification", source) for source in [left, right]]
joined, _, audit = prepare(paths, {"keys": ["key"], "join": "outer"})
assert len(joined) == 5 and audit[0]["keys"]["both"] == 1
if (ROOT / "data" / "Etching.csv").is_file():
    wafer = run("eda", {"options": {"x": "Temp_Etching"}}, ["data:Etching.csv"])[1]
    assert wafer["rows"] > 1000
assert client.get(project + "/report").status_code == 200
with patch("app.report_draft", return_value={"kpi": "Project report"}) as draft:
    assert client.post(project + "/report/draft").json["kpi"] == "Project report"
    assert any(row["metrics"].get("검정") == "welch" for row in draft.call_args.args[0]["approved_runs"])
snapshot = client.post(project + "/report", data={"kpi": "Project report"})
assert snapshot.status_code == 303 and b"Project report" in client.get(snapshot.location).data
assert client.get("/p/another/runs/" + reg).status_code == 404
print("End-to-end verification passed.")
