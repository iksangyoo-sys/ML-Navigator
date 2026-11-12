"""7단계: 탐색적 분석."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

from . import blocks as B
from .charts import PALETTE, style


def sample(df: pd.DataFrame, n: int = 2000, seed: int = 42) -> tuple[pd.DataFrame, bool]:
    if len(df) <= n:
        return df, False
    return df.sample(n=n, random_state=seed), True


def num_cols(df, exclude=()):
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]) and c not in exclude]


def cat_cols(df, exclude=(), max_levels=200):
    out = []
    for c in df.columns:
        if c in exclude:
            continue
        s = df[c]
        if not pd.api.types.is_numeric_dtype(s) or s.nunique() <= 10:
            if s.nunique() <= max_levels and not pd.api.types.is_datetime64_any_dtype(s):
                out.append(c)
    return out


def scatter(df, x, y, color=None, trend="ols", n=2000, hover=None):
    d, sampled = sample(df[[c for c in dict.fromkeys([x, y, color] + (hover or [])) if c]].dropna(subset=[x, y]), n)
    if color and d[color].nunique() > 20:
        d = d.copy()
        top = d[color].astype(str).value_counts().head(12).index
        d[color] = d[color].astype(str).where(d[color].astype(str).isin(top), "기타")
    tl = {"ols": "ols", "lowess": "lowess"}.get(trend)
    try:
        fig = px.scatter(d, x=x, y=y, color=(d[color].astype(str) if color and not pd.api.types.is_numeric_dtype(d[color]) else color) if color else None,
                         trendline=tl, render_mode="webgl", hover_data=hover, opacity=0.7, color_discrete_sequence=PALETTE)
    except Exception:
        fig = px.scatter(d, x=x, y=y, render_mode="webgl", opacity=0.7)
    style(fig, f"{x} vs {y}", x, y, height=460)
    xs, ys = pd.to_numeric(df[x], errors="coerce"), pd.to_numeric(df[y], errors="coerce")
    ok = xs.notna() & ys.notna()
    pr = stats.pearsonr(xs[ok], ys[ok]) if ok.sum() > 2 else (np.nan, np.nan)
    sr = stats.spearmanr(xs[ok], ys[ok]) if ok.sum() > 2 else (np.nan, np.nan)
    slope = np.polyfit(xs[ok], ys[ok], 1)[0] if ok.sum() > 2 else np.nan
    out = [B.metrics([("표본(전체)", int(ok.sum())), ("Pearson r", pr[0]), ("p-value", pr[1]), ("Spearman ρ", sr[0]),
                      ("기울기", slope), ("R²(단순)", pr[0] ** 2 if pr[0] == pr[0] else None)]),
           B.chart(fig, caption=f"전체 {int(ok.sum()):,}행 중 {len(d):,}행 표본으로 그렸습니다." if sampled else None)]
    gap = abs(pr[0] - sr[0]) if pr[0] == pr[0] and sr[0] == sr[0] else 0
    out.append(B.interpret(
        f"Pearson r={pr[0]:.3f}는 직선 관계의 강도, Spearman ρ={sr[0]:.3f}는 순위(단조) 관계의 강도입니다. "
        + ("두 값의 차이가 커서 비선형 관계나 이상치 영향이 의심됩니다. LOWESS 추세선을 확인하세요. " if gap > 0.1 else "")
        + "|r|<0.3 약함, 0.3–0.7 중간, 0.7 이상 강함으로 흔히 읽지만, 상관은 인과가 아니며 제3의 변수가 함께 움직였을 수 있습니다(14단계 참고)."))
    return out


def box(df, cat, num, top_n=20, n=None):
    d = df[[cat, num]].dropna().copy()
    d[cat] = d[cat].astype(str)
    total = d[cat].nunique()
    top = d[cat].value_counts().head(top_n).index
    d = d[d[cat].isin(top)]
    order = d.groupby(cat)[num].median().sort_values().index.tolist()
    fig = px.box(d, x=cat, y=num, points=False, category_orders={cat: order}, color_discrete_sequence=[PALETTE[0]])
    style(fig, f"{cat}별 {num}", cat, num, height=440, legend=False)
    g = d.groupby(cat)[num].agg(["count", "mean", "std", "median", "min", "max"]).reset_index().sort_values("mean", ascending=False)
    out = [B.chart(fig, caption=f"전체 범주 {total}개 중 빈도 상위 {top_n}개" if total > top_n else None),
           B.table(g, f"{cat}별 {num} 요약")]
    out += group_test(d, cat, num)
    return out


def group_test(d, cat, num):
    groups = [g[num].values for _, g in d.groupby(cat) if len(g) >= 2]
    if len(groups) < 2:
        return []
    f, p = stats.f_oneway(*groups)
    kw = stats.kruskal(*groups)
    lev = stats.levene(*groups)
    grand = d[num].mean()
    ssb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    sst = ((d[num] - grand) ** 2).sum()
    eta = ssb / sst if sst else np.nan
    res = pd.DataFrame([
        {"검정": "일원 ANOVA", "통계량": f, "p-value": p},
        {"검정": "Kruskal-Wallis (비모수)", "통계량": kw.statistic, "p-value": kw.pvalue},
        {"검정": "Levene 등분산", "통계량": lev.statistic, "p-value": lev.pvalue},
    ])
    out = [B.table(res, "그룹 차이 검정"), B.metrics([("효과크기 η²", eta), ("해석", "작음" if eta < 0.06 else ("중간" if eta < 0.14 else "큼"))])]
    if len(groups) <= 15:
        try:
            from statsmodels.stats.multicomp import pairwise_tukeyhsd
            tk = pairwise_tukeyhsd(d[num], d[cat])
            t = pd.DataFrame(tk._results_table.data[1:], columns=tk._results_table.data[0])
            t.columns = ["그룹1", "그룹2", "평균차", "보정 p", "하한", "상한", "유의"]
            out.append(B.table(t, "Tukey HSD 사후검정 (모든 쌍 비교)"))
        except Exception:
            pass
    out.append(B.interpret(
        f"ANOVA p={p:.4g}{'로 그룹 평균 차이가 유의합니다' if p < 0.05 else '로 그룹 평균 차이가 유의하지 않습니다'}. "
        f"η²={eta:.3f}는 {num} 변동 중 {cat}로 설명되는 비율입니다. "
        + ("Levene 검정이 유의해 등분산 가정이 깨졌으므로 Kruskal-Wallis 결과나 Welch ANOVA를 우선 보세요. " if lev.pvalue < 0.05 else "")
        + "어떤 그룹끼리 다른지는 Tukey HSD의 '유의=True' 행으로 확인합니다."))
    return out


def correlation(df, cols, method="pearson", target=None):
    d = df[cols].apply(pd.to_numeric, errors="coerce")
    corr = d.corr(method=method)
    fig = go.Figure(go.Heatmap(z=corr.values, x=cols, y=cols, zmin=-1, zmax=1, colorscale="RdBu_r",
                               text=np.round(corr.values, 2), texttemplate="%{text}" if len(cols) <= 20 else None))
    style(fig, f"상관계수 히트맵 ({method})", height=max(420, 28 * len(cols) + 140), legend=False)
    out = [B.chart(fig)]
    # 강한 쌍
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append({"변수1": cols[i], "변수2": cols[j], "상관계수": corr.iloc[i, j]})
    pt = pd.DataFrame(pairs)
    if not pt.empty:
        pt = pt.reindex(pt["상관계수"].abs().sort_values(ascending=False).index).head(20)
        out.append(B.table(pt, "상관이 강한 변수 쌍 상위 20"))
    if target and target in cols:
        rows = []
        for c in cols:
            if c == target:
                continue
            ok = d[[c, target]].dropna()
            if len(ok) < 3:
                continue
            if method == "spearman":
                r, p = stats.spearmanr(ok[c], ok[target])
            elif method == "kendall":
                r, p = stats.kendalltau(ok[c], ok[target])
            else:
                r, p = stats.pearsonr(ok[c], ok[target])
            rows.append({"변수명": c, "상관계수": r, "절대상관": abs(r), "p-value": p, "표본": len(ok)})
        tt = pd.DataFrame(rows).sort_values("절대상관", ascending=False)
        out.append(B.table(tt, f"`{target}` 기준 상관 순위", download="target_correlation.csv"))
        bar = go.Figure(go.Bar(y=tt["변수명"][::-1], x=tt["상관계수"][::-1], orientation="h",
                               marker_color=[PALETTE[0] if v > 0 else PALETTE[3] for v in tt["상관계수"][::-1]]))
        style(bar, f"{target}와의 상관계수", "상관계수", None, height=max(300, 22 * len(tt) + 100), legend=False)
        out.append(B.chart(bar))
    strong = pt[pt["상관계수"].abs() >= 0.8] if not pt.empty else pt
    out.append(B.interpret(
        ("특징끼리 |r|≥0.8인 쌍이 있습니다: " + ", ".join(f"{a}–{b}" for a, b in zip(strong["변수1"], strong["변수2"])) +
         ". 선형 모델에서는 계수가 불안정해지므로 VIF를 확인하고 하나를 빼거나 정규화(Ridge)를 쓰세요. " if len(strong) else "")
        + "Pearson은 직선 관계, Spearman은 순위 관계, Kendall은 작은 표본·동순위가 많을 때 적합합니다."))
    return out


def partial_corr(df, x, y, controls):
    import statsmodels.api as sm
    d = df[[x, y] + controls].apply(pd.to_numeric, errors="coerce").dropna()
    C = sm.add_constant(d[controls]) if controls else np.ones((len(d), 1))
    rx = d[x] - sm.OLS(d[x], C).fit().fittedvalues
    ry = d[y] - sm.OLS(d[y], C).fit().fittedvalues
    r, p = stats.pearsonr(rx, ry)
    r0, _ = stats.pearsonr(d[x], d[y])
    return [B.metrics([("단순 상관", r0), ("편상관", r), ("p-value", p), ("표본", len(d))]),
            B.interpret(f"`{', '.join(controls)}`의 영향을 제거하면 {x}–{y} 상관이 {r0:.3f} → {r:.3f}로 바뀝니다. "
                        "크게 줄어들면 두 변수의 관계 상당 부분이 통제변수를 통해 생긴 것입니다.")]


def anova_all(df, target, cats, max_groups=30):
    rows, skipped = [], []
    for c in cats:
        nun = df[c].nunique(dropna=True)
        if nun < 2:
            skipped.append({"범주형 변수": c, "사유": "유효 범주 2개 미만"})
            continue
        if nun > max_groups:
            skipped.append({"범주형 변수": c, "사유": f"범주 수 {nun} > 제한 {max_groups}"})
            continue
        w = df[[c, target]].dropna()
        groups = [g[target].values for _, g in w.groupby(c) if len(g) >= 2]
        if len(groups) < 2:
            skipped.append({"범주형 변수": c, "사유": "표본 2개 이상 그룹 부족"})
            continue
        f, p = stats.f_oneway(*groups)
        kw = stats.kruskal(*groups)
        grand = w[target].mean()
        eta = sum(len(g) * (g.mean() - grand) ** 2 for g in groups) / max(((w[target] - grand) ** 2).sum(), 1e-12)
        rows.append({"범주형 변수": c, "범주 수": nun, "F통계량": f, "p-value": p, "Kruskal p": kw.pvalue, "η²": eta})
    out = []
    if rows:
        t = pd.DataFrame(rows).sort_values("F통계량", ascending=False)
        out.append(B.table(t, f"범주형 변수별 `{target}` 분산분석"))
        out.append(B.interpret("F통계량이 크고 p<0.05인 변수일수록 범주에 따라 타깃 평균이 다릅니다. η²로 효과 크기를 비교하세요. "
                               "개별 변수의 그룹별 분포와 사후검정은 '박스플롯' 탭에서 볼 수 있습니다."))
    else:
        out.append(B.note("계산 가능한 ANOVA 결과가 없습니다.", "warn"))
    if skipped:
        out.append(B.table(pd.DataFrame(skipped), "ANOVA 제외 변수"))
    return out


def cramers_v(ct):
    chi2 = stats.chi2_contingency(ct)[0]
    n = ct.values.sum()
    r, k = ct.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1))) if min(r, k) > 1 else np.nan


def class_target(df, target, nums, cats):
    """범주형 타깃: 범주형 특징은 카이제곱/Cramér's V, 연속형 특징은 클래스별 ANOVA."""
    out = []
    rows = []
    for c in cats:
        if c == target:
            continue
        ct = pd.crosstab(df[c].astype(str), df[target].astype(str))
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            continue
        chi = stats.chi2_contingency(ct)
        rows.append({"변수": c, "범주 수": ct.shape[0], "카이제곱": chi[0], "p-value": chi[1], "Cramér's V": cramers_v(ct)})
    if rows:
        t = pd.DataFrame(rows).sort_values("Cramér's V", ascending=False)
        out.append(B.table(t, "범주형 특징 × 타깃 독립성 검정"))
    rows = []
    for c in nums:
        if c == target:
            continue
        w = df[[c, target]].dropna()
        groups = [g[c].values for _, g in w.groupby(target) if len(g) >= 2]
        if len(groups) < 2:
            continue
        f, p = stats.f_oneway(*groups)
        means = w.groupby(target)[c].mean()
        rows.append({"변수": c, "F통계량": f, "p-value": p, **{f"평균[{k}]": v for k, v in means.items()}})
    if rows:
        t = pd.DataFrame(rows).sort_values("F통계량", ascending=False)
        out.append(B.table(t, "연속형 특징의 클래스별 평균 차이"))
    out.append(B.interpret("Cramér's V는 0(무관)~1(완전 연관)으로 범주형 변수와 타깃의 연관 강도를 나타냅니다(0.1 약함, 0.3 중간, 0.5 강함). "
                           "연속형 특징은 클래스 간 평균 차이가 클수록(F 큼) 분류에 유용할 가능성이 큽니다."))
    return out


def crosstab(df, a, b, normalize="index"):
    ct = pd.crosstab(df[a].astype(str), df[b].astype(str))
    pct = pd.crosstab(df[a].astype(str), df[b].astype(str), normalize=normalize) * 100
    chi = stats.chi2_contingency(ct)
    fig = go.Figure([go.Bar(name=str(col), x=pct.index, y=pct[col]) for col in pct.columns])
    fig.update_layout(barmode="stack")
    style(fig, f"{a}별 {b} 구성비(%)", a, "%", height=400)
    return [B.table(ct.reset_index(), f"교차표 {a} × {b} (빈도)"), B.table(pct.round(2).reset_index(), "행 기준 비율(%)"),
            B.chart(fig), B.metrics([("카이제곱", chi[0]), ("p-value", chi[1]), ("자유도", chi[2]), ("Cramér's V", cramers_v(ct))])]


def vif(df, cols):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    import statsmodels.api as sm
    d = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    d = d.loc[:, d.std() > 0]
    X = sm.add_constant(d)
    rows = [{"변수": c, "VIF": variance_inflation_factor(X.values, i + 1)} for i, c in enumerate(d.columns)]
    t = pd.DataFrame(rows).sort_values("VIF", ascending=False)
    t["판정"] = np.where(t["VIF"] >= 10, "심각", np.where(t["VIF"] >= 5, "주의", "양호"))
    fig = go.Figure(go.Bar(y=t["변수"][::-1], x=t["VIF"][::-1], orientation="h",
                           marker_color=["#e11d48" if v >= 10 else ("#f97316" if v >= 5 else PALETTE[0]) for v in t["VIF"][::-1]]))
    fig.add_vline(x=10, line_dash="dash", line_color="#e11d48")
    style(fig, "분산팽창계수(VIF)", "VIF", None, height=max(300, 22 * len(t) + 100), legend=False)
    return [B.table(t, "다중공선성 진단"), B.chart(fig),
            B.interpret("VIF는 한 변수가 다른 변수들로 얼마나 설명되는지 보여 줍니다. 5 이상 주의, 10 이상 심각으로 봅니다. "
                        "VIF가 높으면 선형·로지스틱 회귀 계수의 부호와 크기가 불안정해지고 인과 해석이 어려워집니다. "
                        "트리 기반 모델의 예측 성능에는 영향이 작지만 변수 중요도가 나뉘어 보일 수 있습니다.")]


def mutual_info(df, target, ttype, features):
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    d = df[features + [target]].dropna(subset=[target]).copy()
    if len(d) > 20000:
        d = d.sample(20000, random_state=0)
    X = pd.DataFrame(index=d.index)
    disc = []
    for c in features:
        s = d[c]
        if pd.api.types.is_numeric_dtype(s):
            X[c] = s.fillna(s.median())
            disc.append(False)
        else:
            X[c] = s.astype(str).factorize()[0]
            disc.append(True)
    y = d[target]
    if ttype == "regression":
        mi = mutual_info_regression(X, pd.to_numeric(y), discrete_features=disc, random_state=0)
    else:
        mi = mutual_info_classif(X, y.astype(str), discrete_features=disc, random_state=0)
    t = pd.DataFrame({"변수": features, "상호정보량": mi}).sort_values("상호정보량", ascending=False)
    fig = go.Figure(go.Bar(y=t["변수"][::-1], x=t["상호정보량"][::-1], orientation="h", marker_color=PALETTE[2]))
    style(fig, f"{target}와의 상호정보량", "MI (nats)", None, height=max(300, 22 * len(t) + 100), legend=False)
    return [B.chart(fig), B.table(t, "상호정보량 순위"),
            B.interpret("상호정보량은 선형·비선형을 가리지 않고 변수가 타깃의 불확실성을 얼마나 줄이는지 측정합니다. "
                        "상관계수는 낮은데 상호정보량이 높은 변수는 비선형 관계일 가능성이 크므로 트리·앙상블 모델에서 효과를 볼 수 있습니다.")]


def scatter_matrix(df, cols, color=None, n=1500):
    d, _ = sample(df[cols + ([color] if color else [])].dropna(), n)
    fig = px.scatter_matrix(d, dimensions=cols, color=d[color].astype(str) if color else None, opacity=0.5,
                            color_discrete_sequence=PALETTE)
    fig.update_traces(diagonal_visible=False, marker={"size": 3})
    style(fig, "산점도 행렬", height=180 * len(cols) + 120)
    return [B.chart(fig)]


def trend(df, time_col, value_cols, freq="D", agg="mean"):
    d = df[[time_col] + value_cols].copy()
    if not pd.api.types.is_datetime64_any_dtype(d[time_col]):
        conv = pd.to_datetime(d[time_col], errors="coerce")
        if conv.notna().mean() > 0.8:
            d[time_col] = conv
    if pd.api.types.is_datetime64_any_dtype(d[time_col]) and freq != "none":
        g = d.set_index(time_col)[value_cols].resample(freq).agg(agg).reset_index()
    else:
        g = d.sort_values(time_col)
    fig = go.Figure()
    for i, c in enumerate(value_cols):
        fig.add_trace(go.Scattergl(x=g[time_col], y=g[c], mode="lines+markers" if len(g) < 200 else "lines", name=c,
                                   line={"color": PALETTE[i % len(PALETTE)]}))
        roll = g[c].rolling(max(3, len(g) // 20), min_periods=1).mean()
        fig.add_trace(go.Scatter(x=g[time_col], y=roll, mode="lines", name=f"{c} 이동평균", line={"dash": "dot", "color": PALETTE[i % len(PALETTE)]}))
    style(fig, "시간 추이", time_col, None, height=420)
    return [B.chart(fig, caption=f"{freq} 단위 {agg} 집계" if freq != "none" else "원자료 순서")]


def related_summary(df, target, ttype, features, top=10):
    """여러 지표를 합쳐 타깃과 관련성이 큰 변수 상위 N개를 요약."""
    rows = []
    y = df[target]
    for c in features:
        s = df[c]
        w = pd.DataFrame({"x": s, "y": y}).dropna()
        if len(w) < 10 or w["x"].nunique() < 2:
            continue
        num = pd.api.types.is_numeric_dtype(s) and s.nunique() > 10
        try:
            if ttype == "regression":
                yy = pd.to_numeric(w["y"], errors="coerce")
                if num:
                    r, p = stats.spearmanr(w["x"], yy)
                    rows.append({"변수": c, "유형": "연속", "지표": "Spearman ρ", "값": r, "강도": abs(r), "p-value": p})
                else:
                    groups = [g.values for _, g in yy.groupby(w["x"].astype(str)) if len(g) >= 2]
                    f, p = stats.f_oneway(*groups)
                    grand = yy.mean()
                    eta = sum(len(g) * (g.mean() - grand) ** 2 for g in groups) / max(((yy - grand) ** 2).sum(), 1e-12)
                    rows.append({"변수": c, "유형": "범주", "지표": "η (상관비)", "값": np.sqrt(eta), "강도": np.sqrt(eta), "p-value": p})
            else:
                if num:
                    groups = [g.values for _, g in w["x"].groupby(w["y"].astype(str)) if len(g) >= 2]
                    f, p = stats.f_oneway(*groups)
                    xx = w["x"]
                    grand = xx.mean()
                    eta = sum(len(g) * (g.mean() - grand) ** 2 for g in groups) / max(((xx - grand) ** 2).sum(), 1e-12)
                    rows.append({"변수": c, "유형": "연속", "지표": "η (상관비)", "값": np.sqrt(eta), "강도": np.sqrt(eta), "p-value": p})
                else:
                    ct = pd.crosstab(w["x"].astype(str), w["y"].astype(str))
                    v = cramers_v(ct)
                    rows.append({"변수": c, "유형": "범주", "지표": "Cramér's V", "값": v, "강도": v, "p-value": stats.chi2_contingency(ct)[1]})
        except Exception:
            continue
    if not rows:
        return [B.note("요약할 변수가 없습니다.", "warn")]
    t = pd.DataFrame(rows).sort_values("강도", ascending=False)
    topt = t.head(top)
    fig = go.Figure(go.Bar(y=topt["변수"][::-1], x=topt["강도"][::-1], orientation="h", marker_color=PALETTE[0],
                           text=[f"{a} {b:.3f}" for a, b in zip(topt["지표"][::-1], topt["값"][::-1])], textposition="auto"))
    style(fig, f"`{target}`와 관련성이 큰 변수 상위 {len(topt)}", "관련성 강도(0~1)", None, height=max(300, 30 * len(topt) + 100), legend=False)
    names = ", ".join(f"**{v}**" for v in topt["변수"].head(5))
    return [B.chart(fig), B.table(t, "변수별 관련성 지표 (전체)"),
            B.interpret(f"타깃과 관련성이 가장 큰 변수는 {names} 순입니다. 연속형은 Spearman 순위상관, 범주형은 상관비 η 또는 Cramér's V로 "
                        "0~1 사이 강도를 맞춰 비교했습니다. 이 순위는 단변량 관계이므로, 다른 변수와 함께 쓸 때의 기여도는 11~12단계 모델 중요도에서 확인하세요.")]
