"""16단계: 신규 데이터 예측, 스키마 검증, 드리프트 진단, 성능 비교."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

from . import blocks as B
from .charts import PALETTE, style
from .modeling import clf_metrics, reg_metrics
from .recipe import apply_for_new_data


def psi(expected: pd.Series, actual: pd.Series, bins=10) -> float:
    e, a = expected.dropna(), actual.dropna()
    if len(e) < 10 or len(a) < 10:
        return np.nan
    if pd.api.types.is_numeric_dtype(e) and e.nunique() > bins:
        edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        ep = np.histogram(e, edges)[0] / len(e)
        ap = np.histogram(a, edges)[0] / len(a)
    else:
        cats = pd.Index(e.astype(str).unique()).union(a.astype(str).unique())
        ep = e.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0).values
        ap = a.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0).values
    ep, ap = np.clip(ep, 1e-4, None), np.clip(ap, 1e-4, None)
    return float(np.sum((ap - ep) * np.log(ap / ep)))


def run(ctx: dict, new_df: pd.DataFrame, info: dict) -> tuple[list, pd.DataFrame]:
    pipe, ttype, classes, recipe, features, target = ctx["pipe"], ctx["ttype"], ctx["classes"], ctx["recipe"], ctx["features"], ctx["target"]
    base_cols, train_base, Xtr = ctx["base_cols"], ctx["base_train"], ctx["Xtr"]
    out = [B.metrics([("신규 행", len(new_df)), ("신규 열", new_df.shape[1]), ("인코딩", info.get("encoding") or "-"),
                      ("타깃 포함", target in new_df.columns)])]
    # 스키마 검증
    missing = [c for c in base_cols if c not in new_df.columns]
    extra = [c for c in new_df.columns if c not in base_cols and c != target]
    rows = []
    for c in base_cols:
        if c not in new_df.columns:
            rows.append({"열": c, "상태": "누락", "내용": "모델 입력에 필요하지만 없습니다. 학습 기준값으로 대체됩니다."})
            continue
        tr, nw = train_base[c], new_df[c]
        if pd.api.types.is_numeric_dtype(tr) != pd.api.types.is_numeric_dtype(nw):
            rows.append({"열": c, "상태": "타입 불일치", "내용": f"학습 {tr.dtype} / 신규 {nw.dtype}"})
        if not pd.api.types.is_numeric_dtype(tr):
            unseen = sorted(set(nw.dropna().astype(str)) - set(tr.dropna().astype(str)))
            if unseen:
                rows.append({"열": c, "상태": "새 범주", "내용": ", ".join(unseen[:10]) + (" …" if len(unseen) > 10 else "") + " → '기타'로 처리"})
        else:
            lo, hi = tr.min(), tr.max()
            oor = float(((nw < lo) | (nw > hi)).mean() * 100)
            if oor > 0:
                rows.append({"열": c, "상태": "학습 범위 밖", "내용": f"{oor:.1f}% 행이 [{lo:.4g}, {hi:.4g}] 밖"})
        mr = float(nw.isna().mean() * 100)
        if mr > float(tr.isna().mean() * 100) + 5:
            rows.append({"열": c, "상태": "결측 증가", "내용": f"학습 {tr.isna().mean() * 100:.1f}% → 신규 {mr:.1f}%"})
    out.append(B.table(pd.DataFrame(rows) if rows else pd.DataFrame([{"열": "-", "상태": "문제 없음", "내용": "모든 입력 열이 학습 스키마와 일치합니다."}]), "스키마 검증"))
    if extra:
        out.append(B.note(f"모델에 쓰지 않는 열 {len(extra)}개는 결과에 그대로 남깁니다: {', '.join(extra[:15])}", "info"))
    # 예측
    df2, errs = apply_for_new_data(new_df, recipe)
    if errs:
        out.append(B.note("파생변수 재적용 경고:\n" + "\n".join("• " + e for e in errs), "warn"))
    for f in features:
        if f not in df2.columns:
            df2[f] = np.nan
    X = df2[features]
    result = new_df.copy()
    if ttype == "regression":
        p = pipe.predict(X)
        result["예측값"] = p
        q = ctx.get("conformal")
        if q:
            result["예측하한_90"] = p - q["90"]
            result["예측상한_90"] = p + q["90"]
    else:
        pr = pipe.predict_proba(X)
        result["예측값"] = [classes[i] for i in pr.argmax(1)]
        for i, c in enumerate(classes):
            result[f"예측확률_{c}"] = pr[:, i]
        result["예측확신도"] = pr.max(1)
    out.append(B.table(result.head(200), "신규 데이터 예측 결과 (상위 200행)", download="new_data_prediction.csv"))
    if ttype == "regression":
        fig = go.Figure(go.Histogram(x=result["예측값"], marker_color=PALETTE[0], name="신규 예측", opacity=0.7, histnorm="probability density"))
        fig.add_trace(go.Histogram(x=ctx["ytr"], marker_color=PALETTE[7], name="학습 타깃", opacity=0.5, histnorm="probability density"))
        fig.update_layout(barmode="overlay")
        style(fig, "신규 예측값 분포 vs 학습 타깃 분포", "값", "밀도", height=340)
        out.append(B.chart(fig))
    else:
        vc = result["예측값"].astype(str).value_counts()
        fig = go.Figure(go.Bar(x=vc.index, y=vc.values, marker_color=PALETTE[0]))
        style(fig, "신규 데이터 예측 클래스 분포", None, "건수", height=300, legend=False)
        out.append(B.chart(fig))
    # 드리프트
    drows = []
    for c in base_cols:
        if c not in new_df.columns:
            continue
        tr, nw = train_base[c], new_df[c]
        v = psi(tr, nw)
        ks_p = None
        if pd.api.types.is_numeric_dtype(tr) and pd.api.types.is_numeric_dtype(nw) and nw.notna().sum() > 5:
            ks_p = float(stats.ks_2samp(tr.dropna(), nw.dropna()).pvalue)
        drows.append({"변수": c, "PSI": v, "KS p-value": ks_p,
                      "학습 평균": float(tr.mean()) if pd.api.types.is_numeric_dtype(tr) else None,
                      "신규 평균": float(pd.to_numeric(nw, errors="coerce").mean()) if pd.api.types.is_numeric_dtype(tr) else None,
                      "판정": "-" if pd.isna(v) else ("큰 변화" if v >= 0.25 else ("약간 변화" if v >= 0.1 else "안정"))})
    if drows:
        dt = pd.DataFrame(drows).sort_values("PSI", ascending=False)
        fig = go.Figure(go.Bar(y=dt["변수"][::-1], x=dt["PSI"][::-1], orientation="h",
                               marker_color=["#e11d48" if (x or 0) >= 0.25 else ("#f97316" if (x or 0) >= 0.1 else PALETTE[2]) for x in dt["PSI"][::-1]]))
        fig.add_vline(x=0.1, line_dash="dot", line_color="#f97316")
        fig.add_vline(x=0.25, line_dash="dash", line_color="#e11d48")
        style(fig, "변수별 분포 변화 (PSI)", "PSI", None, height=max(320, 22 * len(dt) + 100), legend=False)
        out += [B.chart(fig), B.table(dt, "드리프트 진단")]
        big = dt[dt["판정"] == "큰 변화"]["변수"].tolist()
        out.append(B.interpret("PSI(모집단 안정성 지수)는 학습 데이터 대비 분포가 얼마나 달라졌는지 나타냅니다. 0.1 미만 안정, 0.1~0.25 약간 변화, 0.25 이상 큰 변화로 봅니다. "
                               + (f"{', '.join(big)}에서 큰 변화가 있어 예측 신뢰도가 떨어질 수 있습니다. 원료·설비·측정 방법 변경 여부를 확인하고, 필요하면 재학습하세요." if big else "주요 변수의 분포가 안정적이어서 학습 시 성능이 유지될 것으로 기대됩니다.")))
    # 성능 비교
    if target in new_df.columns:
        y = new_df[target]
        ok = y.notna()
        if ttype == "regression":
            m_new = reg_metrics(pd.to_numeric(y[ok]), result.loc[ok, "예측값"])
        else:
            cmap = {str(c): i for i, c in enumerate(classes)}
            yi = y[ok].astype(str).map(cmap)
            ok2 = yi.notna()
            yy = yi[ok2].astype(int).values
            pr = result.loc[ok, [f"예측확률_{c}" for c in classes]].values[ok2.values]
            m_new = clf_metrics(yy, pr.argmax(1), pr, classes)
        m_test = ctx.get("test_metrics", {})
        comp = pd.DataFrame([{"지표": k, "학습 시 평가 데이터": m_test.get(k), "신규 데이터": v,
                              "변화": (v - m_test[k]) if m_test.get(k) is not None and v is not None else None} for k, v in m_new.items()])
        out.append(B.table(comp, "신규 데이터 성능평가 (학습 시 성능과 비교)"))
        main = "R2_Score" if ttype == "regression" else "Accuracy"
        if m_test.get(main) is not None:
            drop = m_test[main] - m_new[main]
            out.append(B.note(f"{main}가 {m_test[main]:.4f} → {m_new[main]:.4f}로 {'하락' if drop > 0 else '상승'}했습니다."
                              + (" 성능 저하가 커서 재학습을 권합니다." if drop > 0.1 else ""), "warn" if drop > 0.1 else "info"))
    return out, result
