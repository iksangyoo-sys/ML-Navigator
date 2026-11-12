"""11~12단계: 모델 비교 시각화와 설명 가능성."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn import metrics as M

from . import blocks as B
from .charts import PALETTE, style


def compare_regression(results: dict, best: str, Xte: pd.DataFrame, yte, max_points=2000) -> list:
    r = results[best]
    y = np.asarray(yte, dtype=float)
    p = np.asarray(r["test_pred"], dtype=float)
    idx = np.arange(len(y))
    if len(y) > max_points:
        idx = np.random.default_rng(0).choice(len(y), max_points, replace=False)
    lo, hi = float(min(y.min(), p.min())), float(max(y.max(), p.max()))
    fig = go.Figure([go.Scattergl(x=y[idx], y=p[idx], mode="markers", marker={"size": 5, "color": PALETTE[0], "opacity": 0.6}, name="평가 데이터"),
                     go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line={"color": PALETTE[3], "dash": "dash"}, name="정확 예측선")])
    style(fig, f"{r['name']} 실제값 vs 예측값", "실제값", "예측값", height=440)
    res = y - p
    rf = go.Figure([go.Scattergl(x=p[idx], y=res[idx], mode="markers", marker={"size": 5, "color": PALETTE[0], "opacity": 0.6}, name="잔차")])
    rf.add_hline(y=0, line_dash="dash", line_color=PALETTE[3])
    style(rf, "잔차 vs 예측값", "예측값", "잔차(실제-예측)", height=380, legend=False)
    hist = go.Figure(go.Histogram(x=res, marker_color=PALETTE[0], nbinsx=50))
    style(hist, "잔차 분포", "잔차", "빈도", height=320, legend=False)
    from scipy import stats
    (osm, osr), (sl, ic, _) = stats.probplot(res, dist="norm")
    qq = go.Figure([go.Scatter(x=osm, y=osr, mode="markers", marker={"size": 4, "color": PALETTE[0]}, name="잔차"),
                    go.Scatter(x=[osm.min(), osm.max()], y=[ic + sl * osm.min(), ic + sl * osm.max()], mode="lines", line={"color": PALETTE[1]}, name="기준선")])
    style(qq, "잔차 Q-Q 플롯", "이론 분위수", "잔차 분위수", height=320, legend=False)
    # 모델별 실제 vs 예측 비교
    allfig = go.Figure()
    for i, (k, rr) in enumerate(results.items()):
        if k == "baseline":
            continue
        allfig.add_trace(go.Scattergl(x=y[idx], y=np.asarray(rr["test_pred"], dtype=float)[idx], mode="markers", name=rr["name"],
                                      marker={"size": 4, "opacity": 0.5, "color": PALETTE[i % len(PALETTE)]}, visible=True if k == best else "legendonly"))
    allfig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line={"color": "#111827", "dash": "dash"}, name="정확 예측선"))
    style(allfig, "모델별 실제값 vs 예측값 (범례를 눌러 비교)", "실제값", "예측값", height=440)
    err = Xte.copy()
    err.insert(0, "실제값", y)
    err.insert(1, "예측값", p)
    err.insert(2, "오차", res)
    err = err.reindex(err["오차"].abs().sort_values(ascending=False).index).head(30)
    het = stats.spearmanr(np.abs(res), p)[0]
    return [B.chart(fig), B.chart(allfig), B.chart(rf), B.chart(hist), B.chart(qq),
            B.table(err, "오차가 큰 사례 상위 30"),
            B.interpret(
                "점이 대각선에 가까울수록 예측이 정확합니다. 잔차 그림에서 점이 0을 중심으로 고르게 흩어져야 좋은 모델입니다. "
                + (f"예측값이 커질수록 잔차의 폭이 커집니다(이분산, ρ={het:.2f}). 타깃 로그 변환을 검토하세요. " if het > 0.2 else "")
                + "잔차에 곡선 패턴이 보이면 모델이 비선형 관계를 놓친 것이고, 특정 구간에 오차가 몰리면 그 구간 데이터를 보강해야 합니다. "
                "오차가 큰 사례 표에서 공통 조건을 찾으면 공정 이상이나 측정 문제의 단서가 됩니다.")]


def compare_classification(results: dict, best: str, yte, classes: list) -> list:
    y = np.asarray(yte)
    k = len(classes)
    out = []
    r = results[best]
    pred = r["test_pred"]
    cm = M.confusion_matrix(y, pred, labels=list(range(k)))
    cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    labels = [str(c) for c in classes]
    hm = go.Figure(go.Heatmap(z=cmn, x=labels, y=labels, colorscale="Blues", zmin=0, zmax=1,
                              text=[[f"{cm[i, j]}<br>({cmn[i, j] * 100:.1f}%)" for j in range(k)] for i in range(k)], texttemplate="%{text}"))
    hm.update_yaxes(autorange="reversed")
    style(hm, f"{r['name']} 혼동행렬 (행=실제, 열=예측)", "예측", "실제", height=380, legend=False)
    out.append(B.chart(hm))
    rep = M.classification_report(y, pred, labels=list(range(k)), target_names=labels, output_dict=True, zero_division=0)
    out.append(B.table(pd.DataFrame(rep).T.reset_index().rename(columns={"index": "클래스"}), "클래스별 성능 리포트"))
    if k == 2:
        roc = go.Figure()
        pr = go.Figure()
        for i, (key, rr) in enumerate(results.items()):
            if rr["test_proba"] is None or key == "baseline":
                continue
            s = rr["test_proba"][:, 1]
            fpr, tpr, _ = M.roc_curve(y, s)
            auc = M.roc_auc_score(y, s)
            roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{rr['name']} AUC={auc:.4f}", line={"color": PALETTE[i % len(PALETTE)], "width": 3 if key == best else 1.5}))
            pp, rc, _ = M.precision_recall_curve(y, s)
            ap = M.average_precision_score(y, s)
            pr.add_trace(go.Scatter(x=rc, y=pp, mode="lines", name=f"{rr['name']} AP={ap:.4f}", line={"color": PALETTE[i % len(PALETTE)]}))
        roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line={"dash": "dash", "color": "#94a3b8"}))
        style(roc, "ROC Curve 비교", "False Positive Rate", "True Positive Rate", height=460)
        pr.add_hline(y=float(np.mean(y)), line_dash="dash", line_color="#94a3b8", annotation_text="무작위 기준")
        style(pr, "Precision-Recall 곡선", "재현율(Recall)", "정밀도(Precision)", height=420)
        out += [B.chart(roc), B.chart(pr)]
        if r["test_proba"] is not None:
            s = r["test_proba"][:, 1]
            out += threshold_blocks(y, s, classes)
            out += calibration_blocks(y, s, r["name"])
            out += gain_blocks(y, s)
    out.append(B.interpret(
        "혼동행렬 대각선이 정답 비율입니다. 대각선 밖의 큰 칸은 모델이 자주 헷갈리는 클래스 쌍입니다. "
        + ("ROC AUC는 임계값과 무관한 순위 능력이고, 양성 비율이 낮을 때는 PR 곡선이 더 현실적인 성능을 보여 줍니다. "
           "임계값 표에서 업무 목적(불량을 놓치지 않기 vs 오경보 줄이기)에 맞는 임계값을 고르면 같은 모델로도 성과가 크게 달라집니다. "
           "보정 곡선이 대각선에 가까우면 예측 확률을 실제 확률로 믿어도 됩니다." if k == 2 else "")))
    return out


def threshold_blocks(y, s, classes) -> list:
    ths = np.linspace(0.02, 0.98, 49)
    rows = []
    for t in ths:
        p = (s >= t).astype(int)
        rows.append({"임계값": t, "정확도": M.accuracy_score(y, p), "정밀도": M.precision_score(y, p, zero_division=0),
                     "재현율": M.recall_score(y, p, zero_division=0), "F1": M.f1_score(y, p, zero_division=0),
                     "특이도": M.recall_score(y, p, pos_label=0, zero_division=0)})
    t = pd.DataFrame(rows)
    t["Youden J"] = t["재현율"] + t["특이도"] - 1
    bf1 = t.loc[t["F1"].idxmax()]
    bj = t.loc[t["Youden J"].idxmax()]
    fig = go.Figure()
    for i, c in enumerate(["정밀도", "재현율", "F1", "정확도"]):
        fig.add_trace(go.Scatter(x=t["임계값"], y=t[c], mode="lines", name=c, line={"color": PALETTE[i]}))
    fig.add_vline(x=float(bf1["임계값"]), line_dash="dot", line_color="#111827", annotation_text=f"F1 최대 {bf1['임계값']:.2f}")
    fig.add_vline(x=0.5, line_dash="dash", line_color="#94a3b8")
    style(fig, f"임계값별 지표 (양성 = {classes[1]})", "임계값", "지표", height=400)
    return [B.chart(fig),
            B.metrics([("기본 임계값", 0.5), ("F1 최대 임계값", float(bf1["임계값"])), ("그때 F1", float(bf1["F1"])),
                       ("Youden 최적 임계값", float(bj["임계값"])), ("그때 재현율", float(bj["재현율"]))], "임계값 최적화"),
            B.table(t.iloc[::4].reset_index(drop=True), "임계값별 지표 (일부)")]


def calibration_blocks(y, s, name) -> list:
    from sklearn.calibration import calibration_curve
    frac, mean = calibration_curve(y, s, n_bins=10, strategy="quantile")
    fig = go.Figure([go.Scatter(x=mean, y=frac, mode="lines+markers", name=name, line={"color": PALETTE[0]}),
                     go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="완벽 보정", line={"dash": "dash", "color": "#94a3b8"})])
    style(fig, "보정 곡선 (예측 확률 vs 실제 비율)", "예측 확률 평균", "실제 양성 비율", height=380)
    return [B.chart(fig), B.metrics([("Brier 점수", M.brier_score_loss(y, s))])]


def gain_blocks(y, s) -> list:
    order = np.argsort(-s)
    ys = np.asarray(y)[order]
    cum = np.cumsum(ys) / max(ys.sum(), 1)
    pct = np.arange(1, len(ys) + 1) / len(ys)
    dec = []
    for d in range(1, 11):
        n = int(len(ys) * d / 10)
        dec.append({"상위(%)": d * 10, "누적 포착률(%)": cum[n - 1] * 100 if n else 0, "리프트": (cum[n - 1] / (d / 10)) if n else 0})
    fig = go.Figure([go.Scatter(x=pct * 100, y=cum * 100, mode="lines", name="모델", line={"color": PALETTE[0]}),
                     go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="무작위", line={"dash": "dash", "color": "#94a3b8"})])
    style(fig, "누적 이득 곡선", "확률 상위 비율(%)", "양성 누적 포착률(%)", height=380)
    return [B.chart(fig), B.table(pd.DataFrame(dec), "십분위 리프트")]


def model_comparison(results: dict, ttype: str, classes: list) -> list:
    rows = []
    for k, r in results.items():
        rows.append({"모델명": r["name"], **r["test"]})
    t = pd.DataFrame(rows)
    out = []
    metrics_ = ["R2_Score", "RMSE", "MAE", "MAPE"] if ttype == "regression" else (
        ["Accuracy", "AUC", "Balanced_Acc", "MCC"] if len(classes) == 2 else ["Accuracy", "F1_macro", "Balanced_Acc", "MCC"])
    metrics_ = [m for m in metrics_ if m in t.columns]
    fig = go.Figure()
    for i, m in enumerate(metrics_):
        fig.add_trace(go.Bar(name=m, x=t["모델명"], y=t[m], marker_color=PALETTE[i]))
    fig.update_layout(barmode="group")
    style(fig, "모델별 평가 지표", None, None, height=420)
    out.append(B.chart(fig))
    box = go.Figure()
    main = "R2_Score" if ttype == "regression" else "Accuracy"
    for i, (k, r) in enumerate(results.items()):
        if main in r["cv"]:
            box.add_trace(go.Box(y=r["cv"][main]["folds"], name=r["name"], marker_color=PALETTE[i % len(PALETTE)], boxpoints="all"))
    style(box, f"교차검증 {main} 분포 (폴드별)", None, main, height=400, legend=False)
    out.append(B.chart(box))
    out.append(B.interpret("교차검증 상자가 좁을수록 데이터가 바뀌어도 성능이 안정적인 모델입니다. 평균이 비슷하면 더 단순하고 안정적인 모델이 운영에 유리합니다."))
    return out


def learning_curve_blocks(pipe, X, y, ttype, cv) -> list:
    from sklearn.model_selection import learning_curve
    scoring = "r2" if ttype == "regression" else "accuracy"
    sizes, tr, va = learning_curve(pipe, X, y, cv=cv, scoring=scoring, train_sizes=np.linspace(0.1, 1.0, 6), n_jobs=-1)
    fig = go.Figure([
        go.Scatter(x=sizes, y=tr.mean(1), mode="lines+markers", name="학습", line={"color": PALETTE[0]}),
        go.Scatter(x=sizes, y=va.mean(1), mode="lines+markers", name="검증", line={"color": PALETTE[1]}),
    ])
    style(fig, f"학습곡선 ({scoring})", "학습 표본 수", scoring, height=380)
    gap = tr.mean(1)[-1] - va.mean(1)[-1]
    slope = va.mean(1)[-1] - va.mean(1)[-2]
    return [B.chart(fig), B.interpret(
        f"학습과 검증 곡선의 마지막 차이는 {gap:.3f}입니다. "
        + ("차이가 커서 과적합 경향이 있습니다. 규제 강화나 데이터 추가가 도움이 됩니다. " if gap > 0.1 else "")
        + ("검증 성능이 아직 올라가는 중이라 데이터를 더 모으면 성능이 좋아질 가능성이 큽니다." if slope > 0.005 else "검증 성능이 거의 수렴해 데이터를 늘려도 개선 폭은 작을 것입니다."))]


# ---------------------------------------------------------------- 설명 가능성


def permutation_blocks(pipe, X, y, ttype, n_repeats=5, seed=0) -> tuple[list, pd.DataFrame]:
    from sklearn.inspection import permutation_importance
    scoring = "r2" if ttype == "regression" else "accuracy"
    Xs, ys = X, y
    if len(X) > 3000:
        Xs = X.sample(3000, random_state=seed)
        ys = pd.Series(np.asarray(y), index=X.index).loc[Xs.index]
    pi = permutation_importance(pipe, Xs, ys, n_repeats=n_repeats, random_state=seed, scoring=scoring, n_jobs=-1)
    t = pd.DataFrame({"변수": X.columns, "중요도(성능 감소)": pi.importances_mean, "표준편차": pi.importances_std})
    t = t.sort_values("중요도(성능 감소)", ascending=False)
    top = t.head(25)
    fig = go.Figure(go.Bar(y=top["변수"][::-1], x=top["중요도(성능 감소)"][::-1], orientation="h",
                           error_x={"type": "data", "array": top["표준편차"][::-1]}, marker_color=PALETTE[0]))
    style(fig, f"순열 중요도 (평가 데이터, {scoring} 감소량)", "성능 감소", None, height=max(320, 24 * len(top) + 100), legend=False)
    return [B.chart(fig), B.table(t, "순열 중요도", download="permutation_importance.csv"),
            B.interpret("순열 중요도는 변수 하나의 값을 무작위로 섞었을 때 성능이 얼마나 떨어지는지로 중요도를 잽니다. "
                        "원래 변수 단위에서 계산되고 모델 종류와 무관하게 비교할 수 있습니다. 서로 상관이 높은 변수들은 서로를 대신할 수 있어 중요도가 낮게 나올 수 있습니다.")], t


def model_importance_blocks(pipe) -> list:
    model = pipe.named_steps["model"]
    try:
        names = list(pipe.named_steps["pre"].get_feature_names_out())
    except Exception:
        return []
    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_)
        label = "트리 기반 중요도(불순도 감소)"
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        imp = np.abs(coef).mean(axis=0) if coef.ndim > 1 else coef
        label = "표준화 계수"
    else:
        return []
    t = pd.DataFrame({"특징": names[:len(imp)], "값": imp})
    t = t.reindex(t["값"].abs().sort_values(ascending=False).index)
    top = t.head(25)
    fig = go.Figure(go.Bar(y=top["특징"][::-1], x=top["값"][::-1], orientation="h",
                           marker_color=[PALETTE[0] if v >= 0 else PALETTE[3] for v in top["값"][::-1]]))
    style(fig, label, None, None, height=max(320, 24 * len(top) + 100), legend=False)
    return [B.chart(fig), B.table(t, label)]


def shap_blocks(pipe, Xtr, Xte, ttype, classes, max_rows=500) -> tuple[list, dict]:
    import shap
    pre = pipe.named_steps["pre"]
    model = pipe.named_steps["model"]
    Pte = pre.transform(Xte.iloc[:max_rows] if len(Xte) > max_rows else Xte)
    Ptr = pre.transform(Xtr.sample(min(200, len(Xtr)), random_state=0))
    names = list(Pte.columns)
    tree_like = type(model).__name__.lower()
    try:
        if any(k in tree_like for k in ("forest", "tree", "boost", "xgb", "lgbm", "gradient")):
            explainer = shap.TreeExplainer(model)
            if "forest" in tree_like or "extratrees" in tree_like:
                Pte = Pte.iloc[:300]  # 깊은 숲 모델은 정확 계산 비용이 커서 표본을 줄인다
            sv = explainer.shap_values(Pte, check_additivity=False)
        elif hasattr(model, "coef_"):
            explainer = shap.LinearExplainer(model, Ptr)
            sv = explainer.shap_values(Pte)
        else:
            f = model.predict if ttype == "regression" else model.predict_proba
            bg = shap.kmeans(Ptr, min(20, len(Ptr))) if len(Ptr) > 20 else Ptr
            explainer = shap.KernelExplainer(f, bg)
            sub = Pte.iloc[:min(100, len(Pte))]
            sv = explainer.shap_values(sub, nsamples=100, silent=True)
            Pte = sub
    except Exception as e:
        return [B.note(f"SHAP 계산에 실패했습니다: {e}", "warn")], {}
    sv = np.asarray(sv) if not isinstance(sv, list) else sv
    if isinstance(sv, list):
        sv = sv[1] if len(sv) == 2 else np.mean(np.abs(np.stack(sv)), axis=0)
    elif sv.ndim == 3:
        sv = sv[:, :, 1] if sv.shape[2] == 2 else np.abs(sv).mean(axis=2)
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(-mean_abs)[:20]
    bar = go.Figure(go.Bar(y=[names[i] for i in order][::-1], x=mean_abs[order][::-1], orientation="h", marker_color=PALETTE[4]))
    style(bar, "SHAP 평균 |기여도|", "평균 |SHAP|", None, height=max(320, 24 * len(order) + 100), legend=False)
    # beeswarm 유사: 변수별 점, 색=특징값 크기
    bee = go.Figure()
    for rank, i in enumerate(order[::-1]):
        vals = Pte.iloc[:, i].values.astype(float)
        rng = np.ptp(vals) or 1
        col = (vals - np.nanmin(vals)) / rng
        jitter = np.random.default_rng(i).normal(0, 0.12, len(vals))
        bee.add_trace(go.Scattergl(x=sv[:, i], y=rank + jitter, mode="markers", showlegend=False,
                                   marker={"size": 4, "color": col, "colorscale": "RdBu_r", "cmin": 0, "cmax": 1,
                                           "showscale": rank == len(order) - 1, "colorbar": {"title": "특징값", "tickvals": [0, 1], "ticktext": ["낮음", "높음"]}},
                                   hovertext=[f"{names[i]}={v:.4g}" for v in vals]))
    bee.update_yaxes(tickvals=list(range(len(order))), ticktext=[names[i] for i in order[::-1]])
    bee.add_vline(x=0, line_color="#94a3b8")
    style(bee, "SHAP 요약 (점=개별 사례, 색=특징값)", "SHAP 값 (예측에 대한 기여)", None, height=max(360, 26 * len(order) + 120), legend=False)
    t = pd.DataFrame({"특징": names, "평균 |SHAP|": mean_abs, "평균 SHAP": sv.mean(axis=0)}).sort_values("평균 |SHAP|", ascending=False)
    dep = []
    for i in order[:3]:
        vals = Pte.iloc[:, i].values.astype(float)
        f = go.Figure(go.Scattergl(x=vals, y=sv[:, i], mode="markers", marker={"size": 5, "color": PALETTE[4], "opacity": 0.6}))
        style(f, f"SHAP 의존성: {names[i]}", f"{names[i]} (전처리 후 값)", "SHAP", height=320, legend=False)
        dep.append(B.chart(f))
    return [B.chart(bee), B.chart(bar), B.table(t, "SHAP 기여도", download="shap_importance.csv")] + dep + [
        B.interpret("SHAP은 각 사례의 예측값을 '평균 예측 + 변수별 기여'로 나눠 보여 줍니다. 요약 그림에서 빨간 점(특징값 높음)이 오른쪽(+)에 몰려 있으면 "
                    "그 변수가 클수록 예측이 커진다는 뜻입니다. 의존성 그림의 곡선 모양으로 비선형 효과와 임계 구간을 확인할 수 있습니다. "
                    "단, SHAP은 '모델이 어떻게 예측하는가'의 설명이며, 변수를 바꾸면 결과가 바뀐다는 인과 효과는 14단계에서 별도로 확인해야 합니다." +
                    (f" (분류 모델은 양성 클래스 `{classes[1]}` 확률 기준)" if ttype != "regression" and len(classes) == 2 else ""))], \
        {"names": names, "mean_abs": mean_abs.tolist()}


def pdp_blocks(pipe, X, features: list[str], ttype, classes) -> list:
    from sklearn.inspection import partial_dependence
    out = []
    Xs = X.sample(min(len(X), 1000), random_state=0)
    for f in features:
        try:
            num = pd.api.types.is_numeric_dtype(Xs[f]) and Xs[f].nunique() > 10
            kw = {"kind": "both", "grid_resolution": 30} if num else {"kind": "average", "categorical_features": [f]}
            pd_ = partial_dependence(pipe, Xs, [f], response_method="auto", **kw)
            grid = pd_["grid_values"][0]
            avg = pd_["average"]
            avg = avg[-1] if avg.ndim > 1 and avg.shape[0] > 1 and ttype != "regression" else avg[0]
            fig = go.Figure()
            if num and "individual" in pd_:
                ind = pd_["individual"]
                ind = ind[-1] if ind.shape[0] > 1 else ind[0]
                for row in ind[np.random.default_rng(0).choice(len(ind), min(40, len(ind)), replace=False)]:
                    fig.add_trace(go.Scatter(x=grid, y=row, mode="lines", line={"color": "rgba(37,99,235,0.12)"}, showlegend=False, hoverinfo="skip"))
            if num:
                fig.add_trace(go.Scatter(x=grid, y=avg, mode="lines", line={"color": PALETTE[1], "width": 3}, name="평균(PDP)"))
            else:
                fig.add_trace(go.Bar(x=[str(g) for g in grid], y=avg, marker_color=PALETTE[1], name="평균(PDP)"))
            ylab = "예측값" if ttype == "regression" else f"P({classes[-1] if len(classes) == 2 else '클래스'})"
            style(fig, f"부분의존도: {f}" + (" (+ICE)" if num else ""), f"{f} (원 단위)", ylab, height=340)
            out.append(B.chart(fig))
        except Exception as e:
            out.append(B.note(f"{f} 부분의존도 계산 실패: {e}", "warn"))
    out.append(B.interpret("부분의존도(PDP)는 다른 변수를 그대로 두고 한 변수만 바꿨을 때 평균 예측이 어떻게 변하는지 원 단위로 보여 줍니다. "
                           "옅은 선(ICE)은 개별 사례별 반응이며, 선들의 모양이 서로 다르면 다른 변수와의 교호작용이 있다는 뜻입니다. "
                           "곡선이 평평해지는 구간은 더 바꿔도 효과가 없는 포화 구간이므로 공정 조건 설정에 참고할 수 있습니다."))
    return out
