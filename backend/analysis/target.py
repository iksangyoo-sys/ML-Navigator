"""4단계: 종속변수 분포 확인."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from . import blocks as B
from .charts import PALETTE, style


def guess_type(s: pd.Series) -> str:
    nun = s.nunique(dropna=True)
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s) and nun > 10:
        return "regression"
    return "binary" if nun == 2 else "multiclass"


TYPE_LABEL = {"regression": "연속형(회귀)", "binary": "범주형(이진 분류)", "multiclass": "범주형(다중 분류)"}


def numeric_blocks(s: pd.Series, name: str, sample_n: int = 5000) -> list:
    s = pd.to_numeric(s, errors="coerce")
    v = s.dropna()
    if v.empty:
        return [B.note("수치값이 없습니다.", "warn")]
    desc = v.describe()
    skew, kurt = float(v.skew()), float(v.kurt())
    fig = make_subplots(rows=2, cols=1, row_heights=[0.22, 0.78], shared_xaxes=True, vertical_spacing=0.03)
    fig.add_trace(go.Box(x=v, name="", marker_color=PALETTE[0], boxmean=True, orientation="h"), 1, 1)
    fig.add_trace(go.Histogram(x=v, nbinsx=min(60, max(10, int(np.sqrt(len(v))))), marker_color=PALETTE[0], opacity=0.85, name="빈도"), 2, 1)
    style(fig, f"{name} 분포 (히스토그램 + 박스플롯)", None, None, height=420, legend=False)
    smp = v.sample(min(len(v), sample_n), random_state=0)
    (osm, osr), (slope, inter, r) = stats.probplot(smp, dist="norm")
    qq = go.Figure([go.Scatter(x=osm, y=osr, mode="markers", marker={"size": 4, "color": PALETTE[0]}, name="표본"),
                    go.Scatter(x=[osm.min(), osm.max()], y=[inter + slope * osm.min(), inter + slope * osm.max()], mode="lines",
                               line={"color": PALETTE[1]}, name="정규 기준선")])
    style(qq, "정규 Q-Q 플롯", "이론 분위수", "표본 분위수", height=380)
    tests = []
    if len(smp) >= 3:
        sh = stats.shapiro(smp)
        tests.append({"검정": "Shapiro-Wilk", "통계량": sh.statistic, "p-value": sh.pvalue, "표본": len(smp)})
    ad = stats.anderson(smp, dist="norm")
    crit = dict(zip(ad.significance_level, ad.critical_values))
    tests.append({"검정": "Anderson-Darling", "통계량": ad.statistic, "p-value": None, "표본": len(smp),
                  "5% 임계값": crit.get(5.0)})
    if len(smp) >= 20:
        nt = stats.normaltest(smp)
        tests.append({"검정": "D'Agostino K²", "통계량": nt.statistic, "p-value": nt.pvalue, "표본": len(smp)})
    q1, q3 = v.quantile([0.25, 0.75])
    iqr = q3 - q1
    n_out = int(((v < q1 - 1.5 * iqr) | (v > q3 + 1.5 * iqr)).sum())
    normal = all((t.get("p-value") is None or t["p-value"] > 0.05) for t in tests)
    suggestion = []
    if skew > 1:
        suggestion.append("오른쪽 꼬리가 깁니다(양의 왜도). log·sqrt 변환이나 Box-Cox를 고려하세요.")
    elif skew < -1:
        suggestion.append("왼쪽 꼬리가 깁니다(음의 왜도). 제곱 변환이나 Yeo-Johnson을 고려하세요.")
    if abs(kurt) > 3:
        suggestion.append("첨도가 커서 극단값이 많습니다. 이상치 점검(6단계)을 권합니다.")
    if v.min() > 0 and v.max() / max(v.min(), 1e-12) > 100:
        suggestion.append("값의 범위가 100배 이상이라 MAPE가 작은 값에 크게 좌우될 수 있습니다.")
    interp = (f"평균 {B.fmt(desc['mean'])}, 중앙값 {B.fmt(desc['50%'])}, 표준편차 {B.fmt(desc['std'])}입니다. "
              f"왜도 {skew:.3f}, 첨도 {kurt:.3f}"
              f"{'로 정규분포에 가깝습니다.' if abs(skew) < 0.5 and abs(kurt) < 1 else '로 정규분포와 차이가 있습니다.'} "
              f"정규성 검정은 {'기각되지 않았습니다(p>0.05).' if normal else '기각되었습니다(p≤0.05). 다만 표본이 크면 작은 차이도 유의하게 나오므로 Q-Q 플롯 모양을 함께 보세요.'} "
              f"IQR 기준 이상값은 {n_out:,}개({n_out / len(v) * 100:.2f}%)입니다. 회귀 모델은 타깃 정규성을 요구하지 않지만, "
              "선형회귀의 잔차 진단과 예측 구간 해석에는 영향을 줍니다.")
    return [
        B.metrics([("유효 표본", len(v)), ("결측", int(s.isna().sum())), ("평균", desc["mean"]), ("표준편차", desc["std"]),
                   ("왜도", skew), ("첨도", kurt)]),
        B.table(desc.to_frame().T.reset_index(drop=True), "기술통계량"),
        B.chart(fig),
        B.chart(qq),
        B.table(pd.DataFrame(tests), "정규성 검정"),
        B.interpret(interp + ("\n\n" + "\n".join("• " + x for x in suggestion) if suggestion else "")),
    ]


def categorical_blocks(s: pd.Series, name: str) -> list:
    vc = s.astype(str).where(s.notna(), "(결측)").value_counts()
    cnt = vc.reset_index()
    cnt.columns = [name, "count"]
    cnt["비율(%)"] = (cnt["count"] / cnt["count"].sum() * 100).round(2)
    bar = go.Figure(go.Bar(x=cnt[name].astype(str), y=cnt["count"], text=cnt["count"], textposition="outside", marker_color=PALETTE[0]))
    style(bar, f"{name} 막대그래프", name, "빈도", height=380, legend=False)
    pie = go.Figure(go.Pie(labels=cnt[name].astype(str), values=cnt["count"], hole=0.35, marker={"colors": PALETTE}))
    style(pie, f"{name} 파이차트", height=380)
    ratio = vc.max() / max(vc.min(), 1)
    nclass = s.nunique(dropna=True)
    msg = (f"클래스 {nclass}개, 가장 많은 클래스와 가장 적은 클래스의 비율은 {ratio:.1f}:1입니다. ")
    if ratio >= 3:
        msg += ("불균형이 있습니다. 정확도(Accuracy)만 보면 오해할 수 있으니 클래스별 재현율·F1, PR-AUC를 함께 보세요. "
                "8단계에서 층화 분할을, 10단계에서 클래스 가중치를, 11단계에서 임계값 조정을 권합니다.")
    else:
        msg += "클래스 비율이 비교적 고릅니다."
    return [B.metrics([("클래스 수", nclass), ("최다 클래스", str(vc.index[0])), ("최소 클래스", str(vc.index[-1])),
                       ("불균형 비율", round(float(ratio), 2))]),
            B.table(cnt, "클래스 빈도"), B.chart(bar), B.chart(pie), B.interpret(msg)]


def analyze(df: pd.DataFrame, target: str, ttype: str) -> list:
    s = df[target]
    head = [B.note(f"종속변수 `{target}` · {TYPE_LABEL[ttype]} 으로 설정했습니다.", "success")]
    if ttype == "regression":
        return head + numeric_blocks(s, target)
    return head + categorical_blocks(s, target)
