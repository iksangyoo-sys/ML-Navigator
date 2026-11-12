"""14단계: DoWhy 인과분석 (Model → Identify → Estimate → Refute)과 보조 추정."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import blocks as B
from .charts import PALETTE, style


def patch_networkx():
    try:
        import networkx as nx
        if not hasattr(nx.algorithms, "d_separated"):
            from networkx.algorithms.d_separation import is_d_separator
            nx.algorithms.d_separated = lambda g, x, y, z: is_d_separator(g, set(x), set(y), set(z))
    except Exception:
        pass


def prepare(df: pd.DataFrame, treatment: str, outcome: str, controls: list[str], ttype: str, classes: list,
            outcome_class: str | None = None, standardize: bool = False, binarize: str | None = None,
            impute: bool = True) -> tuple[pd.DataFrame, dict]:
    """인과분석용 숫자 데이터. 범주형 통제변수는 더미로 바꾼다."""
    cols = list(dict.fromkeys([treatment, outcome] + controls))
    d = df[cols].copy()
    info = {"outcome_label": outcome, "treatment_label": treatment}
    # 결과변수
    if ttype != "regression":
        cls = outcome_class or (classes[-1] if classes else None)
        d[outcome] = (d[outcome].astype(str) == str(cls)).astype(float)
        info["outcome_label"] = f"P({outcome}={cls})"
    else:
        d[outcome] = pd.to_numeric(d[outcome], errors="coerce")
    # 처치변수
    t = d[treatment]
    if not pd.api.types.is_numeric_dtype(t):
        vals = t.astype(str).value_counts().index.tolist()
        ref = vals[0]
        d[treatment] = (t.astype(str) != ref).astype(float) if len(vals) == 2 else pd.factorize(t.astype(str))[0].astype(float)
        info["treatment_note"] = f"범주형 처치: 기준 `{ref}` 대비" if len(vals) == 2 else "범주 코드화(해석 주의)"
    t = pd.to_numeric(d[treatment], errors="coerce")
    if binarize:
        thr = float(t.median()) if binarize == "median" else float(binarize)
        d[treatment] = (t > thr).astype(float)
        info["treatment_note"] = f"처치 이진화: {treatment} > {thr:.4g} 이면 1"
        info["binarized"] = thr
    # 통제변수
    ctrl_cols = []
    for c in controls:
        if c in (treatment, outcome):
            continue
        s = d[c]
        if pd.api.types.is_numeric_dtype(s):
            ctrl_cols.append(c)
        else:
            top = s.astype(str).value_counts().head(8).index[1:]
            for v in top:
                nm = f"{c}={v}"
                d[nm] = (s.astype(str) == v).astype(float)
                ctrl_cols.append(nm)
            d = d.drop(columns=[c])
    d = d.apply(pd.to_numeric, errors="coerce")
    d = d[d[outcome].notna()]
    info["n_imputed"] = 0
    if impute:
        # 처치·통제변수의 남은 결측은 중앙값으로 채워 표본 손실을 막는다(결과변수는 채우지 않음)
        for c in [treatment] + ctrl_cols:
            n = int(d[c].isna().sum())
            if n:
                d[c] = d[c].fillna(d[c].median())
                info["n_imputed"] += n
    d = d.dropna()
    info["n_dropped"] = len(df) - len(d)
    info["t_std"] = float(d[treatment].std())
    if standardize:
        for c in [treatment] + ctrl_cols:
            sd = d[c].std()
            if sd and not set(d[c].unique()) <= {0.0, 1.0}:
                d[c] = (d[c] - d[c].mean()) / sd
        info["scale"] = "표준화 1단위(1 표준편차)"
    else:
        info["scale"] = "원 단위 1단위"
    info["controls"] = ctrl_cols
    return d, info


def build_graph(treatment, outcome, controls, edges: list[list[str]] | None):
    import networkx as nx
    g = nx.DiGraph()
    g.add_nodes_from([treatment, outcome] + controls)
    if edges:
        for a, b in edges:
            if a in g and b in g:
                g.add_edge(a, b)
    else:
        for c in controls:
            g.add_edge(c, treatment)
            g.add_edge(c, outcome)
    g.add_edge(treatment, outcome)
    return g


def graph_figure(g, treatment, outcome) -> go.Figure:
    import networkx as nx
    try:
        pos = nx.spring_layout(g, seed=1, k=1.2)
    except Exception:
        pos = nx.circular_layout(g)
    pos[treatment] = np.array([-1.0, 0.0])
    pos[outcome] = np.array([1.0, 0.0])
    fig = go.Figure()
    for a, b in g.edges():
        x0, y0 = pos[a]
        x1, y1 = pos[b]
        fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                           arrowhead=3, arrowsize=1.2, arrowwidth=1.5 if not (a == treatment and b == outcome) else 3,
                           arrowcolor="#e11d48" if (a == treatment and b == outcome) else "#94a3b8", standoff=14, startstandoff=14)
    xs, ys, txt, col = [], [], [], []
    for n, (x, y) in pos.items():
        xs.append(x)
        ys.append(y)
        txt.append(n)
        col.append("#e11d48" if n == treatment else ("#2563eb" if n == outcome else "#64748b"))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers+text", text=txt, textposition="top center",
                             marker={"size": 18, "color": col}, hoverinfo="text"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    style(fig, "인과 그래프 (빨강=처치, 파랑=결과, 회색=통제/교란)", height=420, legend=False)
    return fig


def run_dowhy(d: pd.DataFrame, info: dict, treatment: str, outcome: str, method: str, edges=None,
              refuters=("random_common_cause", "placebo_treatment_refuter", "data_subset_refuter"), seed=0) -> dict:
    patch_networkx()
    from dowhy import CausalModel
    controls = info["controls"]
    if len(d) < 30:
        raise ValueError("인과분석에 사용할 유효 데이터가 30행 미만입니다.")
    if edges:
        g = build_graph(treatment, outcome, controls, edges)
        model = CausalModel(data=d, treatment=treatment, outcome=outcome, graph=g)
    else:
        model = CausalModel(data=d, treatment=treatment, outcome=outcome, common_causes=controls or None)
    est_id = model.identify_effect(proceed_when_unidentifiable=True)
    kwargs = {}
    if method.startswith("backdoor.propensity"):
        if set(d[treatment].unique()) - {0.0, 1.0}:
            raise ValueError("성향점수 방법은 0/1 처치변수만 가능합니다. '처치 이진화' 옵션을 사용하세요.")
        kwargs["target_units"] = "ate"
    est = model.estimate_effect(est_id, method_name=method, confidence_intervals=method == "backdoor.linear_regression",
                                test_significance=method == "backdoor.linear_regression", **kwargs)
    refs = {}
    labels = {"random_common_cause": "무작위 공통원인 추가", "placebo_treatment_refuter": "위약(placebo) 처치",
              "data_subset_refuter": "데이터 부분집합(80%)", "bootstrap_refuter": "부트스트랩",
              "add_unobserved_common_cause": "미관측 교란 민감도"}
    for r in refuters:
        kw = {"method_name": r, "random_seed": seed}
        if r in ("random_common_cause", "placebo_treatment_refuter", "data_subset_refuter", "bootstrap_refuter"):
            kw["num_simulations"] = 30
        if r == "placebo_treatment_refuter":
            kw["placebo_type"] = "permute"
        if r == "data_subset_refuter":
            kw["subset_fraction"] = 0.8
        if r == "add_unobserved_common_cause":
            kw.update({"confounders_effect_on_treatment": "linear", "confounders_effect_on_outcome": "linear",
                       "effect_strength_on_treatment": 0.05, "effect_strength_on_outcome": 0.05})
            kw.pop("random_seed", None)
        try:
            rr = model.refute_estimate(est_id, est, **kw)
            refs[labels.get(r, r)] = rr
        except Exception as e:
            refs[labels.get(r, r)] = f"검증 실패: {e}"
    return {"model": model, "estimand": est_id, "estimate": est, "refutations": refs, "graph": build_graph(treatment, outcome, controls, edges)}


def refute_table(refs: dict, effect: float) -> pd.DataFrame:
    rows = []
    for name, r in refs.items():
        if isinstance(r, str):
            rows.append({"검증": name, "원래 효과": effect, "검증 후 효과": None, "p-value": None, "판정": r})
            continue
        new = float(np.atleast_1d(r.new_effect)[0]) if r.new_effect is not None else None
        p = None
        try:
            p = r.refutation_result.get("p_value") if isinstance(r.refutation_result, dict) else None
        except Exception:
            pass
        if "위약" in name:
            ok = new is not None and abs(new) < 0.2 * abs(effect) + 1e-9
            verdict = "통과(위약 효과≈0)" if ok else "주의(위약에도 효과)"
        elif "민감도" in name:
            verdict = "참고(약한 미관측 교란 시 효과)"
        else:
            ok = new is not None and abs(new - effect) <= 0.2 * abs(effect) + 1e-9
            verdict = "통과(효과 유지)" if ok else "주의(효과 변동)"
        rows.append({"검증": name, "원래 효과": effect, "검증 후 효과": new, "p-value": p, "판정": verdict})
    return pd.DataFrame(rows)


def ols_effect(d: pd.DataFrame, treatment: str, outcome: str, controls: list[str]):
    import statsmodels.api as sm
    X = sm.add_constant(d[[treatment] + controls], has_constant="add")
    fit = sm.OLS(d[outcome], X).fit(cov_type="HC3")
    ci = fit.conf_int().loc[treatment]
    return fit, float(fit.params[treatment]), float(ci[0]), float(ci[1]), float(fit.pvalues[treatment])


def dml_effect(d: pd.DataFrame, treatment: str, outcome: str, controls: list[str], folds=5, seed=0) -> dict:
    """이중 머신러닝(부분선형 모델): 랜덤포레스트로 Y·T를 통제변수에서 교차 적합 잔차화 후 회귀."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_predict, KFold
    import statsmodels.api as sm
    W = d[controls].values if controls else np.zeros((len(d), 1))
    cv = KFold(folds, shuffle=True, random_state=seed)
    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=seed, n_jobs=-1)
    y_res = d[outcome].values - cross_val_predict(rf, W, d[outcome].values, cv=cv)
    t_res = d[treatment].values - cross_val_predict(rf, W, d[treatment].values, cv=cv)
    fit = sm.OLS(y_res, sm.add_constant(t_res)).fit(cov_type="HC3")
    ci = fit.conf_int()[1]
    return {"effect": float(fit.params[1]), "lo": float(ci[0]), "hi": float(ci[1]), "p": float(fit.pvalues[1])}


def single_blocks(res: dict, d: pd.DataFrame, info: dict, treatment: str, outcome: str, method: str, extra: dict) -> list:
    est = res["estimate"]
    eff = float(np.atleast_1d(est.value)[0])
    ci = None
    try:
        c = est.get_confidence_intervals()
        ci = (float(np.ravel(c)[0]), float(np.ravel(c)[1]))
    except Exception:
        pass
    unit = info["scale"]
    out = [
        B.metrics([("추정 인과효과", eff), ("95% 신뢰구간", f"{ci[0]:.4g} ~ {ci[1]:.4g}" if ci else "-"),
                   ("사용 데이터 행 수", len(d)), ("통제변수 수", len(info["controls"])), ("효과 단위", unit)]),
        B.chart(graph_figure(res["graph"], treatment, outcome)),
        B.heading("1단계 Model", f"처치 `{treatment}` → 결과 `{info['outcome_label']}` 인과효과를 가정했습니다."),
        B.text(("통제(교란) 변수: " + ", ".join(info["controls"])) if info["controls"] else "통제변수 없음 (교란이 없다고 가정)"),
        B.heading("2단계 Identify", "선택한 통제변수로 뒷문(backdoor) 경로를 차단해 효과를 식별합니다."),
        B.code(str(res["estimand"])),
        B.heading("3단계 Estimate", method),
        B.code(str(est)),
        B.heading("4단계 Refute", "효과가 우연이나 모형 설정 때문인지 검증합니다."),
        B.table(refute_table(res["refutations"], eff), "반박 검정 결과"),
    ]
    with_text = [B.code(str(r), name) for name, r in res["refutations"].items()]
    out.append(B.tabs([("검증 상세 출력", with_text)]))
    comp = [{"방법": f"DoWhy {method}", "효과": eff, "하한": ci[0] if ci else None, "상한": ci[1] if ci else None}]
    if "ols" in extra:
        o = extra["ols"]
        comp.append({"방법": "통제변수 OLS (HC3 강건 표준오차)", "효과": o[1], "하한": o[2], "상한": o[3], "p-value": o[4]})
    if "dml" in extra:
        m = extra["dml"]
        comp.append({"방법": "이중 머신러닝 DML (랜덤포레스트)", "효과": m["effect"], "하한": m["lo"], "상한": m["hi"], "p-value": m["p"]})
    if "naive" in extra:
        comp.append({"방법": "단순 상관(통제 없음, 참고)", "효과": extra["naive"]})
    ct = pd.DataFrame(comp)
    fig = go.Figure()
    for i, r in ct.iterrows():
        err = None
        if r.get("하한") is not None and not pd.isna(r.get("하한")):
            err = {"type": "data", "symmetric": False, "array": [r["상한"] - r["효과"]], "arrayminus": [r["효과"] - r["하한"]]}
        fig.add_trace(go.Scatter(x=[r["효과"]], y=[r["방법"]], mode="markers", marker={"size": 12, "color": PALETTE[i % len(PALETTE)]}, error_x=err, showlegend=False))
    fig.add_vline(x=0, line_dash="dash", line_color="#94a3b8")
    style(fig, "추정 방법별 효과 비교 (95% 신뢰구간)", f"효과 ({unit})", None, height=300, legend=False)
    out += [B.table(ct, "추정 방법 비교"), B.chart(fig)]
    sign = "증가" if eff > 0 else "감소"
    orig = ""
    if info["scale"].startswith("표준화") and info.get("t_std"):
        orig = f" 원 단위로 환산하면 {treatment} 1단위당 약 {eff / info['t_std']:.4g}입니다."
    note_bin = f" ({info['treatment_note']})" if info.get("treatment_note") else ""
    passed = all("통과" in str(v) or "참고" in str(v) for v in refute_table(res["refutations"], eff)["판정"])
    out.append(B.interpret(
        f"다른 통제변수가 같다면 `{treatment}`가 1단위 늘 때{note_bin} `{info['outcome_label']}`가 평균 {abs(eff):.4g}만큼 {sign}한다고 추정됩니다.{orig} "
        + (f"95% 신뢰구간이 0을 {'포함하지 않아' if ci and (ci[0] > 0 or ci[1] < 0) else '포함해'} 효과가 통계적으로 {'유의합니다' if ci and (ci[0] > 0 or ci[1] < 0) else '불확실합니다'}. " if ci else "")
        + ("반박 검정을 모두 통과해 결과가 비교적 견고합니다. " if passed else "일부 반박 검정에서 효과가 흔들려 해석에 주의가 필요합니다. ")
        + "이 해석은 '처치와 결과 모두에 영향을 주는 변수를 모두 통제했다'는 가정 위에서만 성립합니다. 처치 이후에 결정되는 변수(매개·결과 변수)를 통제하면 오히려 편향이 생기니 제외하세요."))
    return out


def batch_effects(d_full: pd.DataFrame, candidates: list[str], outcome: str, ttype: str, classes: list, importance: dict | None,
                  outcome_class=None, standardize=True, progress=None) -> list:
    """모든 후보 변수에 대해 나머지를 통제한 OLS 효과를 계산하고 모델 중요도와 비교."""
    rows = []
    for i, t in enumerate(candidates):
        if progress:
            progress(i / max(len(candidates), 1), f"{t} 효과 추정 중")
        controls = [c for c in candidates if c != t]
        try:
            d, info = prepare(d_full, t, outcome, controls, ttype, classes, outcome_class, standardize=standardize)
            if len(d) < 30:
                continue
            fit, eff, lo, hi, p = ols_effect(d, t, outcome, info["controls"])
            raw = eff / info["t_std"] if standardize and info.get("t_std") else eff
            rows.append({"변수": t, "인과효과(표준화)": eff if standardize else None, "원 단위 효과": raw if standardize else eff,
                         "하한": lo, "상한": hi, "p-value": p, "유의": p < 0.05})
        except Exception:
            continue
    if not rows:
        return [B.note("계산 가능한 변수가 없습니다.", "warn")]
    t = pd.DataFrame(rows)
    key = "인과효과(표준화)" if standardize else "원 단위 효과"
    t = t.reindex(t[key].abs().sort_values(ascending=False).index)
    if importance:
        t["모델 중요도"] = t["변수"].map(importance)
        t["중요도 순위"] = t["모델 중요도"].rank(ascending=False)
        t["효과 순위"] = t[key].abs().rank(ascending=False)
    fig = go.Figure(go.Scatter(x=t[key], y=t["변수"], mode="markers", marker={"size": 10, "color": np.where(t["유의"], PALETTE[3], "#94a3b8")},
                               error_x={"type": "data", "symmetric": False, "array": t["상한"] - t[key], "arrayminus": t[key] - t["하한"]}))
    fig.add_vline(x=0, line_dash="dash", line_color="#94a3b8")
    style(fig, f"변수별 인과효과 ({'1 표준편차당' if standardize else '원 단위'}, 빨강=유의)", "효과", None, height=max(320, 26 * len(t) + 120), legend=False)
    out = [B.chart(fig), B.table(t, "변수 일괄 인과효과 (나머지 변수 통제, HC3)", download="causal_effects.csv")]
    if importance and "모델 중요도" in t:
        sc = go.Figure(go.Scatter(x=t["모델 중요도"], y=t[key].abs(), mode="markers+text", text=t["변수"], textposition="top center",
                                  marker={"size": 10, "color": PALETTE[0]}))
        style(sc, "예측 중요도 vs 인과효과 크기", "모델 순열 중요도", "|인과효과|", height=420, legend=False)
        out.append(B.chart(sc))
        hi_imp_low_eff = t[(t["중요도 순위"] <= 5) & (~t["유의"])]["변수"].tolist()
        out.append(B.interpret(
            "예측에 중요한 변수와 실제로 바꾸면 결과가 바뀌는 변수는 다를 수 있습니다. 예측 중요도는 '결과와 함께 움직이는 정도'이고, "
            "인과효과는 '다른 조건이 같을 때 그 변수만 바꾼 효과'입니다. "
            + (f"{', '.join(hi_imp_low_eff)}은(는) 예측 중요도는 높지만 다른 변수를 통제하면 효과가 유의하지 않습니다. 다른 변수의 대리지표일 가능성이 있어 조정 대상으로 삼기 전에 확인이 필요합니다. " if hi_imp_low_eff else "")
            + "현장에서 조정할 변수를 고를 때는 인과효과가 크고 유의한 변수를 우선하세요."))
    return out


def heterogeneity(d_full: pd.DataFrame, treatment: str, outcome: str, controls: list[str], segment: str, ttype, classes,
                  outcome_class=None, standardize=False, bins=4) -> list:
    seg = d_full[segment]
    if pd.api.types.is_numeric_dtype(seg) and seg.nunique() > bins:
        labels = pd.qcut(seg, q=bins, duplicates="drop").astype(str)
    else:
        labels = seg.astype(str)
    rows = []
    for lab in labels.dropna().unique():
        sub = d_full[labels == lab]
        try:
            ctl = [c for c in controls if c != segment]
            d, info = prepare(sub, treatment, outcome, ctl, ttype, classes, outcome_class, standardize)
            if len(d) < 20:
                continue
            _, eff, lo, hi, p = ols_effect(d, treatment, outcome, info["controls"])
            rows.append({"구간": lab, "표본": len(d), "효과": eff, "하한": lo, "상한": hi, "p-value": p})
        except Exception:
            continue
    if not rows:
        return [B.note("구간별 효과를 계산할 수 없습니다.", "warn")]
    t = pd.DataFrame(rows).sort_values("구간")
    fig = go.Figure(go.Bar(x=t["구간"], y=t["효과"], marker_color=PALETTE[0],
                           error_y={"type": "data", "symmetric": False, "array": t["상한"] - t["효과"], "arrayminus": t["효과"] - t["하한"]}))
    fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
    style(fig, f"`{segment}` 구간별 `{treatment}` 효과", segment, "효과", height=380, legend=False)
    spread = t["효과"].max() - t["효과"].min()
    return [B.chart(fig), B.table(t, "구간별 이질적 효과"),
            B.interpret(f"구간별 효과의 최대·최소 차이는 {spread:.4g}입니다. 신뢰구간이 겹치지 않을 정도로 차이가 크면 `{segment}` 조건에 따라 "
                        f"`{treatment}` 조정 효과가 다르므로, 구간별로 다른 운전 기준을 고려할 수 있습니다.")]
