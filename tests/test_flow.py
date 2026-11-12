"""1~16단계 종단 테스트: Flask 테스트 클라이언트로 전체 흐름을 실행한다.

사용: python -m tests.test_flow [regression|binary|multiclass]
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("ANALYSIS_HOME", tempfile.mkdtemp(prefix="mlweb_test_"))
os.environ.pop("APP_PASSWORD", None)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from backend.app import app  # noqa: E402

c = app.test_client()
FAIL = []


def call(method, url, full=False, **kw):
    r = getattr(c, method)(url, **kw)
    j = r.get_json()
    if not j or not j.get("ok"):
        raise AssertionError(f"{method.upper()} {url} → {r.status_code} {j}")
    return j if full else j.get("data")


def run(pid, name, params=None, allow_fail=False):
    jid = call("post", f"/api/projects/{pid}/run/{name}", json=params or {})["job"]
    t0 = time.time()
    while True:
        j = call("get", f"/api/jobs/{jid}")
        if j["status"] in ("done", "failed", "cancelled"):
            break
        if time.time() - t0 > 900:
            raise TimeoutError(name)
        time.sleep(0.2)
    dt = time.time() - t0
    if j["status"] != "done":
        msg = f"{name}: {j['status']} {j.get('error')}"
        print("  ✗", msg)
        if not allow_fail:
            FAIL.append(msg)
        return None
    kinds = {}
    for b in j["blocks"]:
        kinds[b["type"]] = kinds.get(b["type"], 0) + 1
    print(f"  ✓ {name:16s} {dt:6.1f}s  {kinds}")
    return j["blocks"]


def make_data(kind: str, tmp: Path) -> tuple[list[Path], str]:
    rng = np.random.default_rng(0)
    n = 1500
    if kind == "regression":
        return [ROOT / "data" / "M5탭밀도_예측.csv"], "Final_Tap_Density"
    temp = rng.normal(850, 30, n)
    time_ = rng.uniform(8, 20, n)
    line = rng.choice(["L1", "L2", "L3"], n)
    hum = rng.normal(30, 5, n)
    lot = [f"LOT{i // 10:04d}" for i in range(n)]
    score = 0.04 * (temp - 850) + 0.3 * (time_ - 14) + (line == "L3") * 1.2 - 0.05 * (hum - 30) + rng.normal(0, 1, n)
    a = pd.DataFrame({"LOT": lot, "온도": temp, "시간": time_, "라인": line})
    a.loc[rng.choice(n, 60, replace=False), "시간"] = np.nan
    if kind == "binary":
        y = np.where(score > 0.8, "불량", "양품")
    else:
        y = np.select([score < -0.5, score < 1.0], ["A", "B"], "C")
    b = pd.DataFrame({"LOT": lot, "습도": hum, "판정": y}).drop_duplicates("LOT").reset_index(drop=True)
    # 행 단위 결합 테스트를 위해 b도 LOT 단위 1:1로 만든다 → 실제로 N:1 결합
    b = pd.DataFrame({"LOT": sorted(set(lot))})
    b["습도"] = rng.normal(30, 5, len(b))
    a["판정"] = y
    pa, pb = tmp / f"{kind}_공정.csv", tmp / f"{kind}_환경.xlsx"
    a.to_csv(pa, index=False, encoding="cp949")
    b.to_excel(pb, index=False)
    return [pa, pb], "판정"


def main(kind="regression"):
    tmp = Path(tempfile.mkdtemp())
    files, target = make_data(kind, tmp)
    print(f"\n=== {kind} · {', '.join(f.name for f in files)} ===")
    pid = call("post", "/api/projects", json={"name": f"테스트 {kind}"})["id"]
    for f in files:
        with open(f, "rb") as fh:
            call("post", f"/api/projects/{pid}/upload", data={"files": (fh, f.name)}, content_type="multipart/form-data")
    s = call("get", f"/api/projects/{pid}")
    srcs = [x["id"] for x in s["sources"]]
    run(pid, "s1_profile")
    if len(srcs) > 1:
        run(pid, "s2_keys")
        run(pid, "s2_merge", {"mode": "cols", "sources": srcs, "keys": [["LOT"]], "how": "left"})
    else:
        run(pid, "s2_merge", {"mode": "single", "sources": srcs})
    cols = [x["name"] for x in call("get", f"/api/projects/{pid}")["columns"]]
    nums = [x["name"] for x in call("get", f"/api/projects/{pid}")["columns"] if x["numeric"] and x["name"] != target]
    a, b = nums[0], nums[1]
    run(pid, "s3_derive", {"op": {"op": "arith", "left": a, "right": b, "operator": "/", "new": "비율_ab"}})
    run(pid, "s3_derive", {"op": {"op": "bin", "source": a, "n_bins": 4, "method": "quantile", "new": f"{a}_구간"}})
    run(pid, "s3_derive", {"op": {"op": "transform", "source": b, "kind": "yeojohnson", "new": f"{b}_yj"}})
    run(pid, "s3_tukey", {"column": a})
    run(pid, "s4_target", {"target": target})
    run(pid, "s5_overview")
    miss_cols = [x["name"] for x in call("get", f"/api/projects/{pid}")["columns"] if x["missing"] > 0 and x["name"] != target]
    if miss_cols:
        run(pid, "s5_apply", {"columns": miss_cols[:1], "method": "median"})
        run(pid, "s5_apply", {"columns": miss_cols[:1], "method": "indicator"})
    run(pid, "s6_detect", {"column": a, "method": "iqr"})
    run(pid, "s6_apply", {"columns": [a], "method": "iqr", "action": "winsor"})
    run(pid, "s6_multi", {"columns": nums[:4]})
    s = call("get", f"/api/projects/{pid}")
    cats = [x["name"] for x in s["columns"] if not x["numeric"] and x["role"] == "feature"]
    run(pid, "s7", {"kind": "summary"})
    run(pid, "s7", {"kind": "scatter", "x": a, "y": b, "trend": "lowess"})
    run(pid, "s7", {"kind": "corr", "columns": nums[:8] + ([target] if kind == "regression" else [])})
    run(pid, "s7", {"kind": "partial", "x": a, "y": b, "controls": nums[2:4]})
    if cats:
        run(pid, "s7", {"kind": "box", "cat": cats[0], "num": a})
        run(pid, "s7", {"kind": "crosstab", "a": cats[0], "b": cats[-1]})
    run(pid, "s7", {"kind": "anova" if kind == "regression" else "classtarget"})
    run(pid, "s7", {"kind": "vif"})
    run(pid, "s7", {"kind": "mi"})
    run(pid, "s7", {"kind": "matrix", "columns": nums[:3]})
    run(pid, "s8_split", {"method": "stratified" if kind != "regression" else "random", "test_size": 0.3})
    run(pid, "s9_preprocess", {"scaler": "standard", "top_n": 5})
    s = call("get", f"/api/projects/{pid}")
    ttype = s["target_type"]
    models = ["linear", "rf", "xgb", "ridge"] if ttype == "regression" else ["logit", "rf", "xgb", "hgb"]
    run(pid, "s10_train", {"models": models, "tune": False})
    run(pid, "s10_train", {"models": models[:2], "tune": True, "n_iter": 4})
    run(pid, "s10_train", {"models": models, "tune": False})
    run(pid, "s11_compare", {"learning_curve": True})
    run(pid, "s11_explain", {"kinds": ["perm", "model", "shap", "pdp"]})
    spec = call("get", f"/api/projects/{pid}/input-spec")["spec"]
    row = call("get", f"/api/projects/{pid}/sample-row")["row"]
    run(pid, "s13_predict", {"row": row})
    num_in = [x["name"] for x in spec if x["type"] == "number"]
    run(pid, "s13_whatif", {"row": row, "var": num_in[0]})
    run(pid, "s13_sensitivity", {"row": row})
    treat = num_in[0]
    ctrls = [x for x in num_in[1:6]]
    run(pid, "s14_causal", {"treatment": treat, "controls": ctrls, "method": "backdoor.linear_regression"})
    run(pid, "s14_causal", {"treatment": treat, "controls": ctrls, "method": "backdoor.propensity_score_weighting", "binarize": "median",
                            "refuters": ["placebo_treatment_refuter"]})
    run(pid, "s14_batch", {"candidates": num_in[:8]})
    if cats:
        run(pid, "s14_hetero", {"treatment": treat, "controls": ctrls, "segment": num_in[1]})
    classes = s.get("classes")
    rule = ({"direction": "이상", "target_value": float(np.nanpercentile([float(x) for x in pd.read_csv(files[0])[target].dropna()], 70))}
            if ttype == "regression" else {"target_class": classes[0] if ttype == "binary" else classes[-1], "min_prob": 0.5})
    run(pid, "s15_cf", {"row": row, "mutable": num_in[:5], **rule, "n_samples": 2000}, allow_fail=False)
    run(pid, "s15_cf", {"row": row, "mutable": num_in[:5], **rule, "method": "genetic", "n_samples": 2000,
                        "constraints": {num_in[0]: {"direction": "increase", "cost": 2}}})
    run(pid, "s15_batch", {"mutable": num_in[:5], **rule, "n_rows": 20, "n_samples": 300})
    run(pid, "s15_optimize", {"mutable": num_in[:3], **rule, "n_samples": 100})
    # 16단계: 원본 일부를 신규 데이터로
    src = pd.read_csv(files[0], encoding="cp949" if kind != "regression" else "utf-8-sig")
    if kind != "regression":
        env = pd.read_excel(files[1])
        src = src.merge(env, on="LOT", how="left")
    newp = tmp / "new.csv"
    src.sample(200, random_state=1).to_csv(newp, index=False)
    with open(newp, "rb") as fh:
        nd = call("post", f"/api/projects/{pid}/newdata", data={"file": (fh, "new.csv")}, content_type="multipart/form-data", full=True)["added"]
    run(pid, "s16_predict", {"file": nd["id"]})
    r = c.get(f"/api/projects/{pid}/report/steps")
    assert r.status_code == 200 and len(r.data) > 10000, "보고서 생성 실패"
    print(f"  ✓ 단계별 보고서 {len(r.data) / 1e6:.1f}MB")
    f = call("get", f"/api/projects/{pid}/report/executive-fields")
    e = call("post", f"/api/projects/{pid}/report/executive", json=f)
    print(f"  ✓ 경영 요약 {len(e['html']) / 1e3:.0f}KB")
    s = call("get", f"/api/projects/{pid}")
    stale = [k for k, v in s["results"].items() if v["stale"]]
    print("  stale:", stale)


if __name__ == "__main__":
    kinds = sys.argv[1:] or ["regression", "binary", "multiclass"]
    for k in kinds:
        main(k)
    print("\n실패:" if FAIL else "\n모든 단계 통과", FAIL or "")
    sys.exit(1 if FAIL else 0)
