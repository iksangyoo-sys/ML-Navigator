"""13단계: 직접 입력 예측, What-if 민감도, 개별 예측 설명."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import blocks as B
from .charts import PALETTE, style
from .recipe import apply_for_new_data


def input_spec(base: pd.DataFrame, cols: list[str]) -> list[dict]:
    spec = []
    for c in cols:
        s = base[c]
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 2 and not pd.api.types.is_bool_dtype(s):
            v = pd.to_numeric(s, errors="coerce").dropna()
            spec.append({"name": c, "type": "number", "min": float(v.min()), "max": float(v.max()), "median": float(f"{v.median():.6g}"),
                         "q01": float(v.quantile(0.01)), "q99": float(v.quantile(0.99)), "std": float(v.std() or 0),
                         "integer": bool(np.all(np.mod(v.values, 1) == 0))})
        else:
            vc = s.dropna().value_counts()
            opts = [x.item() if hasattr(x, "item") else x for x in vc.index[:200]]
            spec.append({"name": c, "type": "select", "options": [str(o) for o in opts], "raw_options": opts,
                         "default": str(opts[0]) if opts else ""})
    return spec


def coerce_row(row: dict, spec: list[dict]) -> dict:
    out = {}
    for s in spec:
        v = row.get(s["name"])
        if s["type"] == "number":
            try:
                out[s["name"]] = float(v) if v not in (None, "") else s["median"]
            except (TypeError, ValueError):
                out[s["name"]] = s["median"]
        else:
            # 원래 타입(숫자 범주 등)을 복원
            raw = dict(zip(s["options"], s["raw_options"]))
            out[s["name"]] = raw.get(str(v), v if v not in (None, "") else (s["raw_options"][0] if s["raw_options"] else None))
    return out


def make_X(rows: pd.DataFrame, recipe: list[dict], features: list[str]) -> pd.DataFrame:
    df, _ = apply_for_new_data(rows, recipe)
    for f in features:
        if f not in df.columns:
            df[f] = np.nan
    return df[features]


def shap_rows(pipe, Xbg: pd.DataFrame, Xrows: pd.DataFrame, ttype: str):
    """개별 행에 대한 SHAP 값(전처리 후 특징 기준)과 기준값."""
    import shap
    pre, model = pipe.named_steps["pre"], pipe.named_steps["model"]
    P = pre.transform(Xrows)
    bg = pre.transform(Xbg.sample(min(100, len(Xbg)), random_state=0))
    name = type(model).__name__.lower()
    if any(k in name for k in ("forest", "tree", "boost", "xgb", "lgbm", "gradient")):
        ex = shap.TreeExplainer(model)
        sv = ex.shap_values(P)
        base = ex.expected_value
    elif hasattr(model, "coef_"):
        ex = shap.LinearExplainer(model, bg)
        sv = ex.shap_values(P)
        base = ex.expected_value
    else:
        f = model.predict if ttype == "regression" else model.predict_proba
        ex = shap.KernelExplainer(f, shap.kmeans(bg, min(10, len(bg))) if len(bg) > 10 else bg)
        sv = ex.shap_values(P, nsamples=100, silent=True)
        base = ex.expected_value
    if isinstance(sv, list):
        sv = sv[1] if len(sv) == 2 else sv[0]
        base = np.atleast_1d(base)[1 if len(np.atleast_1d(base)) == 2 else 0]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1] if sv.shape[2] == 2 else sv[:, :, 0]
        base = np.atleast_1d(base)[1 if len(np.atleast_1d(base)) == 2 else 0]
    base = float(np.atleast_1d(base)[0]) if np.ndim(base) else float(base)
    return sv, base, list(P.columns), P


def waterfall(sv_row, names, base, P_row, title) -> dict:
    """개별 예측 기여도: 기준값 대비 변수별 SHAP 기여(빨강=올림, 파랑=내림)."""
    order = np.argsort(-np.abs(sv_row))[:12]
    rest = float(np.sum(sv_row) - np.sum(sv_row[order]))
    labels = [f"{names[i]} = {P_row[i]:.3g}" for i in order]
    vals = [float(sv_row[i]) for i in order]
    if abs(rest) > 1e-12:
        labels.append("기타 변수 합")
        vals.append(rest)
    labels, vals = labels[::-1], vals[::-1]
    fig = go.Figure(go.Bar(y=labels, x=vals, orientation="h", marker_color=["#e11d48" if v > 0 else "#2563eb" for v in vals],
                           text=[f"{v:+.4g}" for v in vals], textposition="outside"))
    fig.add_vline(x=0, line_color="#94a3b8")
    pred = base + float(np.sum(sv_row))
    style(fig, f"{title} · 기준값 {base:.4g} → 예측 {pred:.4g}", "기여도 (모델 출력 단위)", None,
          height=max(340, 28 * len(labels) + 120), legend=False)
    return B.chart(fig, caption="전처리 후 특징 기준입니다(연속형은 표준화된 값).")


def predict_one(ctx: dict, row: dict) -> list:
    pipe, ttype, classes = ctx["pipe"], ctx["ttype"], ctx["classes"]
    spec, recipe, features = ctx["spec"], ctx["recipe"], ctx["features"]
    vals = coerce_row(row, spec)
    X = make_X(pd.DataFrame([vals]), recipe, features)
    out = []
    pred = pipe.predict(X)
    warns = []
    for s in spec:
        if s["type"] == "number":
            v = vals[s["name"]]
            if v < s["min"] or v > s["max"]:
                warns.append(f"`{s['name']}`={v:g}는 학습 범위 [{s['min']:g}, {s['max']:g}] 밖입니다. 외삽 예측은 신뢰도가 낮습니다.")
    if ttype == "regression":
        yhat = float(pred[0])
        items = [("예측값", yhat)]
        q = ctx.get("conformal")
        if q:
            items += [("90% 예측구간", f"{yhat - q['90']:.4g} ~ {yhat + q['90']:.4g}"), ("95% 예측구간", f"{yhat - q['95']:.4g} ~ {yhat + q['95']:.4g}")]
        out.append(B.metrics(items, f"{ctx['model_name']} 예측 결과"))
    else:
        proba = pipe.predict_proba(X)[0]
        lab = classes[int(pred[0])]
        out.append(B.metrics([("예측 클래스", str(lab))] + [(f"P({c})", float(p)) for c, p in zip(classes, proba)], f"{ctx['model_name']} 예측 결과"))
        fig = go.Figure(go.Bar(x=[str(c) for c in classes], y=proba, marker_color=PALETTE[:len(classes)], text=[f"{p:.1%}" for p in proba], textposition="outside"))
        style(fig, "클래스별 예측 확률", None, "확률", height=300, legend=False)
        fig.update_yaxes(range=[0, 1.1])
        out.append(B.chart(fig))
    if warns:
        out.append(B.note("\n".join("• " + w for w in warns), "warn"))
    # 파생변수 포함 최종 입력
    out.append(B.table(X, "모델에 들어간 특징값 (파생변수 자동 계산 포함)"))
    # 다른 모델 비교
    others = []
    for key, (name, p) in ctx.get("others", {}).items():
        try:
            if ttype == "regression":
                others.append({"모델": name, "예측값": float(p.predict(X)[0])})
            else:
                pr = p.predict_proba(X)[0]
                others.append({"모델": name, "예측 클래스": str(classes[int(np.argmax(pr))]), **{f"P({c})": float(v) for c, v in zip(classes, pr)}})
        except Exception:
            pass
    if others:
        out.append(B.table(pd.DataFrame(others), "다른 모델의 예측 비교"))
    try:
        sv, base, names, P = shap_rows(pipe, ctx["Xtr"], X, ttype)
        out.append(waterfall(sv[0], names, base, P.iloc[0].values.astype(float), "이 예측의 변수별 기여도 (SHAP)"))
        out.append(B.interpret("빨간 막대는 예측을 올린 변수, 파란 막대는 내린 변수입니다. 기준값(학습 데이터 평균 예측)에서 출발해 "
                               "각 변수의 기여를 더하면 이 사례의 예측이 됩니다." + (" 분류 모델은 확률의 로그오즈 또는 확률 단위입니다." if ttype != "regression" else "")))
    except Exception as e:
        out.append(B.note(f"개별 SHAP 계산 생략: {e}", "info"))
    return out


def whatif(ctx: dict, row: dict, var: str, points: int = 40) -> list:
    pipe, ttype, classes, spec = ctx["pipe"], ctx["ttype"], ctx["classes"], ctx["spec"]
    vals = coerce_row(row, spec)
    sp = next(s for s in spec if s["name"] == var)
    if sp["type"] == "number":
        grid = np.linspace(sp["min"], sp["max"], points)
    else:
        grid = sp["raw_options"][:30]
    rows = pd.DataFrame([{**vals, var: g} for g in grid])
    X = make_X(rows, ctx["recipe"], ctx["features"])
    if ttype == "regression":
        y = pipe.predict(X)
        ylab = "예측값"
    else:
        pr = pipe.predict_proba(X)
        ci = len(classes) - 1 if len(classes) == 2 else int(np.argmax(pipe.predict_proba(make_X(pd.DataFrame([vals]), ctx["recipe"], ctx["features"]))[0]))
        y = pr[:, ci]
        ylab = f"P({classes[ci]})"
    fig = go.Figure()
    if sp["type"] == "number":
        fig.add_trace(go.Scatter(x=grid, y=y, mode="lines", line={"color": PALETTE[0], "width": 3}, name=ylab))
        fig.add_vline(x=vals[var], line_dash="dash", line_color=PALETTE[3], annotation_text="현재값")
        fig.add_vrect(x0=sp["q01"], x1=sp["q99"], fillcolor="rgba(16,185,129,0.07)", line_width=0, annotation_text="학습 1~99%", annotation_position="top left")
    else:
        fig.add_trace(go.Bar(x=[str(g) for g in grid], y=y, marker_color=PALETTE[0]))
    style(fig, f"What-if: {var} 변화에 따른 {ylab}", var, ylab, height=380, legend=False)
    rng = float(np.max(y) - np.min(y))
    return [B.chart(fig), B.metrics([("예측 최소", float(np.min(y))), ("예측 최대", float(np.max(y))), ("변화 폭", rng),
                                     ("최대가 되는 값", str(grid[int(np.argmax(y))])), ("최소가 되는 값", str(grid[int(np.argmin(y))]))]),
            B.interpret(f"다른 입력은 현재값으로 고정하고 `{var}`만 바꿨을 때 모델 예측의 변화입니다. 모델의 반응일 뿐 실제 인과 효과와는 다를 수 있으며, "
                        "녹색 구간(학습 데이터의 1~99% 범위) 밖은 근거 데이터가 적어 신뢰하기 어렵습니다.")]


def sensitivity_all(ctx: dict, row: dict) -> list:
    """모든 수치 입력을 학습 범위 1~99% 내에서 움직였을 때 예측 변화 폭(토네이도 차트)."""
    pipe, ttype, classes, spec = ctx["pipe"], ctx["ttype"], ctx["classes"], ctx["spec"]
    vals = coerce_row(row, spec)
    base_X = make_X(pd.DataFrame([vals]), ctx["recipe"], ctx["features"])

    def f(X):
        if ttype == "regression":
            return pipe.predict(X)
        return pipe.predict_proba(X)[:, -1]
    base = float(f(base_X)[0])
    rows = []
    for s in spec:
        if s["type"] != "number":
            continue
        r = pd.DataFrame([{**vals, s["name"]: s["q01"]}, {**vals, s["name"]: s["q99"]}])
        y = f(make_X(r, ctx["recipe"], ctx["features"]))
        rows.append({"변수": s["name"], "하위 1% 값": s["q01"], "상위 99% 값": s["q99"], "하위일 때": float(y[0]), "상위일 때": float(y[1]),
                     "변화폭": float(abs(y[1] - y[0]))})
    if not rows:
        return []
    t = pd.DataFrame(rows).sort_values("변화폭", ascending=True)
    fig = go.Figure([go.Bar(y=t["변수"], x=t["하위일 때"] - base, orientation="h", name="하위 1%", marker_color=PALETTE[0]),
                     go.Bar(y=t["변수"], x=t["상위일 때"] - base, orientation="h", name="상위 99%", marker_color=PALETTE[1])])
    fig.update_layout(barmode="overlay")
    style(fig, f"토네이도 민감도 (현재 예측 {base:.4g} 기준 변화)", "예측 변화", None, height=max(320, 26 * len(t) + 120))
    return [B.chart(fig), B.table(t.sort_values("변화폭", ascending=False), "변수별 민감도")]
