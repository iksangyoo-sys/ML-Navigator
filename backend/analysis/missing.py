"""5단계: 결측치 진단과 처리."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

from . import blocks as B
from .charts import style


def overview(df: pd.DataFrame, target: str | None, ttype: str | None) -> list:
    miss = df.isna()
    tab = pd.DataFrame({"변수명": df.columns, "타입": [str(t) for t in df.dtypes], "결측수": miss.sum().values,
                        "결측비율(%)": (miss.mean() * 100).round(2).values})
    tab = tab.sort_values("결측수", ascending=False)
    has = tab[tab["결측수"] > 0]
    out = [B.metrics([("전체 행", len(df)), ("결측 있는 열", len(has)), ("결측 있는 행", int(miss.any(axis=1).sum())),
                      ("완전한 행 비율(%)", round(float((~miss.any(axis=1)).mean() * 100), 2))]),
           B.table(tab, "변수별 결측 수·비율", download="missing_summary.csv")]
    if has.empty:
        out.append(B.note("결측이 있는 열이 없습니다. 이 단계는 건너뛰어도 됩니다.", "success"))
        return out
    bar = go.Figure(go.Bar(x=has["변수명"], y=has["결측비율(%)"], marker_color="#e11d48"))
    style(bar, "결측 비율", None, "%", height=320, legend=False)
    out.append(B.chart(bar))
    cols = has["변수명"].tolist()[:40]
    smp = miss[cols].sample(min(len(df), 400), random_state=0).sort_index()
    heat = go.Figure(go.Heatmap(z=smp.values.astype(int).T, x=list(range(len(smp))), y=cols, colorscale=[[0, "#e2e8f0"], [1, "#e11d48"]],
                                showscale=False))
    style(heat, "결측 패턴 매트릭스 (표본 400행, 빨강=결측)", "행(표본)", None, height=max(260, 22 * len(cols) + 80), legend=False)
    out.append(B.chart(heat))
    if len(cols) >= 2:
        co = miss[cols].astype(int).corr().fillna(0)
        hm = go.Figure(go.Heatmap(z=co.values, x=cols, y=cols, zmin=-1, zmax=1, colorscale="RdBu_r", text=np.round(co.values, 2), texttemplate="%{text}"))
        style(hm, "결측 동시 발생 상관 (1에 가까우면 함께 빔)", height=max(320, 26 * len(cols) + 100), legend=False)
        out.append(B.chart(hm))
    # 결측 여부와 타깃 관련성
    if target and target in df.columns:
        rows = []
        y = df[target]
        for c in cols:
            if c == target:
                continue
            m = miss[c]
            if m.sum() < 3 or (~m).sum() < 3:
                continue
            if ttype == "regression":
                a, b = pd.to_numeric(y[m], errors="coerce").dropna(), pd.to_numeric(y[~m], errors="coerce").dropna()
                if len(a) >= 3 and len(b) >= 3:
                    t = stats.mannwhitneyu(a, b)
                    rows.append({"변수": c, "결측 시 타깃 평균": a.mean(), "관측 시 타깃 평균": b.mean(), "검정": "Mann-Whitney U", "p-value": t.pvalue})
            else:
                ct = pd.crosstab(m, y)
                if ct.shape[0] == 2 and ct.shape[1] >= 2:
                    chi = stats.chi2_contingency(ct)
                    rows.append({"변수": c, "검정": "카이제곱", "통계량": chi[0], "p-value": chi[1]})
        if rows:
            rt = pd.DataFrame(rows)
            out.append(B.table(rt, "결측 여부와 타깃의 관련성"))
            sig = rt[rt["p-value"] < 0.05]["변수"].tolist()
            out.append(B.interpret(
                ("결측 여부에 따라 타깃이 유의하게 다른 변수: " + ", ".join(f"`{s}`" for s in sig) +
                 ". 이런 결측은 무작위가 아닐 가능성이 크므로 결측 지시변수를 추가하면 정보를 보존할 수 있습니다. "
                 if sig else "결측 여부와 타깃 사이에 유의한 관련이 보이지 않습니다(완전 무작위 결측에 가까움). ") +
                "결측 행 삭제는 표본을 줄이고 편향을 만들 수 있으니 결측률이 낮을 때만 권합니다. "
                "결측률이 40%를 넘는 변수는 대체보다 제외나 지시변수 활용을 먼저 검토하세요."))
    return out


def recommend(df: pd.DataFrame) -> list[dict]:
    out = []
    for c in df.columns:
        r = df[c].isna().mean()
        if r == 0:
            continue
        num = pd.api.types.is_numeric_dtype(df[c])
        if r > 0.4:
            rec = "제외 검토 또는 결측 지시변수"
        elif r < 0.02:
            rec = "결측 행 삭제 또는 중앙값/최빈값"
        elif num:
            rec = "중앙값(왜도 큼) 또는 KNN/반복 대체"
        else:
            rec = "최빈값 또는 '결측' 범주"
        out.append({"변수": c, "결측률(%)": round(r * 100, 2), "권장": rec})
    return out
