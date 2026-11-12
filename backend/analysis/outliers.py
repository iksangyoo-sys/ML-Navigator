"""6단계: 이상치 탐지와 처리."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import blocks as B
from .charts import PALETTE, style
from .recipe import bounds

METHOD_LABEL = {"iqr": "IQR", "zscore": "Z-score", "mad": "수정 Z(MAD)", "percentile": "백분위", "custom": "직접 범위"}


def detect(df: pd.DataFrame, col: str, method: str = "iqr", factor: float | None = None, lower=None, upper=None,
           lower_pct: float = 1, upper_pct: float = 99) -> tuple[list, dict]:
    s = pd.to_numeric(df[col], errors="coerce")
    v = s.dropna()
    lo, hi = bounds(v, method, factor, lower, upper, lower_pct, upper_pct)
    mask = (s < lo) | (s > hi)
    n = int(mask.sum())
    q1, q3 = v.quantile([0.25, 0.75])
    fig = make_subplots(rows=2, cols=1, row_heights=[0.25, 0.75], shared_xaxes=True, vertical_spacing=0.03)
    fig.add_trace(go.Box(x=v, name="", marker_color=PALETTE[0], orientation="h", boxpoints="outliers"), 1, 1)
    fig.add_trace(go.Histogram(x=v, marker_color=PALETTE[0], opacity=0.85, nbinsx=60), 2, 1)
    for x, name in [(lo, "하한"), (hi, "상한")]:
        if np.isfinite(x):
            fig.add_vline(x=x, line_dash="dash", line_color="#e11d48", annotation_text=f"{name} {x:.4g}", row="all", col=1)
    style(fig, f"{col} 이상치 확인 ({METHOD_LABEL[method]})", None, None, height=400, legend=False)
    # 방법별 비교
    comp = []
    for m, f in [("iqr", 1.5), ("iqr", 3.0), ("zscore", 3.0), ("mad", 3.5), ("percentile", None)]:
        a, b = bounds(v, m, f)
        k = int(((v < a) | (v > b)).sum())
        comp.append({"방법": f"{METHOD_LABEL[m]}{'' if f is None else f' ({f})'}", "하한": a, "상한": b, "이상치 수": k,
                     "비율(%)": round(k / max(len(v), 1) * 100, 2)})
    info = {"lower": lo, "upper": hi, "count": n}
    blocks = [
        B.metrics([("Q1", q1), ("Q3", q3), ("IQR", q3 - q1), ("하한", lo), ("상한", hi), ("이상치 수", n),
                   ("이상치 비율(%)", round(n / max(len(v), 1) * 100, 2))]),
        B.chart(fig),
        B.table(pd.DataFrame(comp), "탐지 방법별 비교"),
    ]
    if n:
        blocks.append(B.table(df.loc[mask].head(50), f"이상치 행 (최대 50행)"))
    blocks.append(B.interpret(
        "IQR(1.5)은 분포 모양에 관계없이 쓰기 쉽지만 치우친 분포에서는 한쪽 꼬리를 과하게 잡습니다. "
        "Z-score는 정규분포를 가정하고, 극단값 자체가 평균·표준편차를 끌어당기는 약점이 있습니다. "
        "수정 Z(MAD)는 중앙값 기반이라 극단값에 강건합니다. 이상치를 지우기 전에 측정 오류인지, 실제로 일어난 공정 이벤트인지 먼저 확인하세요. "
        "실제 이벤트라면 삭제보다 윈저화(경계값 대체)나 플래그가 정보를 덜 잃습니다."))
    return blocks, info


def multivariate(df: pd.DataFrame, cols: list[str], method: str = "iforest", contamination: float = 0.02) -> list:
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    ok = X.dropna()
    if len(ok) < 20:
        raise ValueError("결측 없는 행이 20개 미만입니다.")
    if method == "iforest":
        from sklearn.ensemble import IsolationForest
        m = IsolationForest(contamination=contamination, random_state=0).fit(ok)
        score = -m.score_samples(ok)
        flag = m.predict(ok) == -1
        label = "Isolation Forest 이상 점수"
    else:
        from sklearn.covariance import MinCovDet
        from scipy import stats
        mcd = MinCovDet(random_state=0).fit(ok)
        score = mcd.mahalanobis(ok)
        thr = stats.chi2.ppf(1 - contamination, df=len(cols))
        flag = score > thr
        label = "강건 마할라노비스 거리²"
    res = ok.copy()
    res["이상점수"] = score
    res["이상여부"] = flag
    fig = go.Figure(go.Histogram(x=score, marker_color=PALETTE[0], nbinsx=60))
    style(fig, label + " 분포", label, "빈도", height=320, legend=False)
    blocks = [B.metrics([("분석 행", len(ok)), ("다변량 이상치", int(flag.sum())), ("비율(%)", round(flag.mean() * 100, 2))]),
              B.chart(fig)]
    if len(cols) >= 2:
        sc = go.Figure(go.Scattergl(x=ok[cols[0]], y=ok[cols[1]], mode="markers",
                                    marker={"color": np.where(flag, "#e11d48", "#94a3b8"), "size": 5}))
        style(sc, f"{cols[0]} vs {cols[1]} (빨강=다변량 이상치)", cols[0], cols[1], height=380, legend=False)
        blocks.append(B.chart(sc))
    blocks.append(B.table(res.sort_values("이상점수", ascending=False).head(50).reset_index().rename(columns={"index": "행번호"}),
                          "이상 점수 상위 50행"))
    blocks.append(B.interpret("다변량 이상치는 변수 하나씩 보면 정상인데 조합이 특이한 행입니다(예: 온도는 높은데 압력은 낮은 경우). "
                              "공정 데이터에서는 센서 오류나 비정상 운전 조건의 단서가 됩니다. 여기서는 탐지만 하며, 제거하려면 단변량 처리나 행 필터를 사용하세요."))
    return blocks


def compare_after(before: pd.Series, after: pd.Series, col: str) -> list:
    fig = go.Figure()
    fig.add_trace(go.Box(y=pd.to_numeric(before, errors="coerce"), name="처리 전", marker_color=PALETTE[7]))
    fig.add_trace(go.Box(y=pd.to_numeric(after, errors="coerce"), name="처리 후", marker_color=PALETTE[0]))
    style(fig, f"{col} 처리 전후 비교", None, col, height=360)
    return [B.chart(fig)]
