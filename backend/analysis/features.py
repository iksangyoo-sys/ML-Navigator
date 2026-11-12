"""3단계: 파생변수. 모든 연산은 레시피로 저장되어 신규 데이터에 그대로 재적용된다."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import blocks as B

OPS = {
    "bin": "구간화(binning)", "dummy": "더미(dummy)", "tukey": "Tukey 거듭제곱", "arith": "사칙연산",
    "transform": "변환(transform)", "date": "날짜 분해", "groupagg": "그룹 집계", "condition": "조건식",
    "lump": "희소 범주 병합", "expr": "수식",
}


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def fit_op(df: pd.DataFrame, op: dict) -> dict:
    """데이터에서 필요한 파라미터를 적합해 op에 저장한다 (생성 시 1회)."""
    op = dict(op)
    kind = op["op"]
    if kind == "bin":
        s = _num(df[op["source"]])
        n = int(op.get("n_bins", 3))
        if op.get("method") == "quantile":
            _, edges = pd.qcut(s, q=n, retbins=True, duplicates="drop")
        else:
            _, edges = pd.cut(s, bins=n, retbins=True, duplicates="drop")
        edges = list(map(float, edges))
        edges[0], edges[-1] = -np.inf, np.inf
        op["edges"] = edges
    elif kind == "dummy":
        op["categories"] = sorted(df[op["source"]].dropna().astype(str).unique().tolist())
    elif kind == "transform":
        s = _num(df[op["source"]]).dropna()
        t = op["kind"]
        if t in ("zscore", "standardize"):
            op["mean"], op["std"] = float(s.mean()), float(s.std(ddof=0) or 1.0)
        elif t == "minmax":
            op["min"], op["max"] = float(s.min()), float(s.max())
        elif t == "boxcox":
            shift = float(max(0, -s.min() + 1e-6)) if s.min() <= 0 else 0.0
            _, lam = stats.boxcox(s + shift)
            op["lambda"], op["shift"] = float(lam), shift
        elif t == "yeojohnson":
            _, lam = stats.yeojohnson(s)
            op["lambda"] = float(lam)
    elif kind == "groupagg":
        g = df.groupby(op["group"])[op["source"]].agg(op.get("agg", "mean"))
        op["mapping"] = {str(k): (None if pd.isna(v) else float(v)) for k, v in g.items()}
    elif kind == "lump":
        vc = df[op["source"]].astype(str).value_counts()
        op["keep"] = vc.head(int(op.get("top_n", 10))).index.tolist()
    elif kind == "date":
        d = pd.to_datetime(df[op["source"]], errors="coerce")
        op["ref"] = d.min().isoformat() if d.notna().any() else None
    return op


def apply_op(df: pd.DataFrame, op: dict) -> pd.DataFrame:
    df = df.copy()
    kind = op["op"]
    new = op.get("new")
    if kind != "expr" and kind not in ("arith",) and op.get("source") and op["source"] not in df.columns:
        raise ValueError(f"열 `{op['source']}`이(가) 없습니다.")
    if kind == "bin":
        s = _num(df[op["source"]])
        cats = pd.cut(s, bins=op["edges"], duplicates="drop")
        df[new] = cats.astype(str).replace("nan", np.nan)
    elif kind == "dummy":
        s = df[op["source"]].astype(str)
        cats = op["categories"][1:] if op.get("drop_first") else op["categories"]
        prefix = op.get("new") or op["source"]
        for c in cats:
            df[f"{prefix}_{c}"] = (s == c).astype(int)
    elif kind == "tukey":
        s = _num(df[op["source"]])
        p = float(op.get("power", 1))
        if p == 0:
            df[new] = np.log1p(s.clip(lower=-0.999999))
        elif p == 1:
            df[new] = s
        elif p < 0:
            df[new] = np.power(s.where(s != 0) + 1e-9, p)
        else:
            df[new] = np.power(s.clip(lower=0) + 1e-9, p) if p != int(p) or p % 2 == 0 else np.power(s, p)
    elif kind == "arith":
        a, b = _num(df[op["left"]]), _num(df[op["right"]]) if op.get("right") in df.columns else float(op.get("constant", 0))
        o = op["operator"]
        if o == "+":
            df[new] = a + b
        elif o == "-":
            df[new] = a - b
        elif o == "*":
            df[new] = a * b
        elif o == "/":
            df[new] = a / (b.replace(0, np.nan) if isinstance(b, pd.Series) else (b or np.nan))
        else:
            raise ValueError("지원하지 않는 연산입니다.")
    elif kind == "transform":
        s = _num(df[op["source"]])
        t = op["kind"]
        if t == "log1p":
            df[new] = np.log1p(s.where(s > -1))
        elif t == "log":
            df[new] = np.log(s.where(s > 0))
        elif t == "sqrt":
            df[new] = np.sqrt(s.clip(lower=0))
        elif t == "square":
            df[new] = s ** 2
        elif t == "reciprocal":
            df[new] = 1 / s.replace(0, np.nan)
        elif t in ("zscore", "standardize"):
            df[new] = (s - op["mean"]) / (op["std"] or 1)
        elif t == "minmax":
            rng = (op["max"] - op["min"]) or 1
            df[new] = (s - op["min"]) / rng
        elif t == "boxcox":
            x = s + op.get("shift", 0)
            lam = op["lambda"]
            df[new] = np.where(x > 0, (np.power(x, lam) - 1) / lam if lam != 0 else np.log(x), np.nan)
        elif t == "yeojohnson":
            vals = s.values.astype(float)
            out = np.full_like(vals, np.nan)
            ok = ~np.isnan(vals)
            out[ok] = stats.yeojohnson(vals[ok], lmbda=op["lambda"])
            df[new] = out
        else:
            raise ValueError("지원하지 않는 변환입니다.")
    elif kind == "date":
        d = pd.to_datetime(df[op["source"]], errors="coerce")
        base = op["source"]
        for part in op.get("parts", ["year", "month", "weekday"]):
            if part == "year":
                df[f"{base}_연"] = d.dt.year
            elif part == "month":
                df[f"{base}_월"] = d.dt.month
            elif part == "day":
                df[f"{base}_일"] = d.dt.day
            elif part == "weekday":
                df[f"{base}_요일"] = d.dt.weekday
            elif part == "hour":
                df[f"{base}_시"] = d.dt.hour
            elif part == "quarter":
                df[f"{base}_분기"] = d.dt.quarter
            elif part == "elapsed" and op.get("ref"):
                df[f"{base}_경과일"] = (d - pd.Timestamp(op["ref"])).dt.total_seconds() / 86400
    elif kind == "groupagg":
        df[new] = df[op["group"]].astype(str).map(op["mapping"]).astype(float)
        if op.get("deviation"):
            df[new] = _num(df[op["source"]]) - df[new]
    elif kind == "condition":
        s = df[op["source"]]
        v = op["value"]
        o = op["operator"]
        sn = _num(s)
        try:
            vn = float(v)
            numeric = sn.notna().any()
        except (TypeError, ValueError):
            vn, numeric = None, False
        if o in (">", ">=", "<", "<=") and numeric:
            mask = {">": sn > vn, ">=": sn >= vn, "<": sn < vn, "<=": sn <= vn}[o]
        elif o == "==":
            mask = (sn == vn) if numeric else (s.astype(str) == str(v))
        elif o == "!=":
            mask = (sn != vn) if numeric else (s.astype(str) != str(v))
        elif o == "in":
            vals = [x.strip() for x in str(v).split(",")]
            mask = s.astype(str).isin(vals)
        else:
            raise ValueError("조건 연산자를 확인하세요.")
        tv, fv = op.get("true_val", 1), op.get("false_val", 0)
        df[new] = np.where(mask, tv, fv)
        try:
            df[new] = pd.to_numeric(df[new])
        except (TypeError, ValueError):
            pass
    elif kind == "lump":
        s = df[op["source"]].astype(str)
        df[new] = s.where(s.isin(op["keep"]), op.get("other", "기타"))
        df.loc[df[op["source"]].isna(), new] = np.nan
    elif kind == "expr":
        df[new] = df.eval(op["expr"], engine="python")
    return df


def apply_all(df: pd.DataFrame, ops: list[dict], strict: bool = False) -> tuple[pd.DataFrame, list[str]]:
    errors = []
    for op in ops:
        try:
            df = apply_op(df, op)
        except Exception as e:  # 신규 데이터에 원천 열이 없을 때 등
            if strict:
                raise
            errors.append(f"{describe(op)}: {e}")
    return df, errors


def describe(op: dict) -> str:
    k = op["op"]
    if k == "bin":
        return f"{op['new']} = {op['source']} {op.get('n_bins')}구간({'분위수' if op.get('method') == 'quantile' else '등간격'})"
    if k == "dummy":
        return f"{op['source']} 더미변수({len(op.get('categories', []))}범주{', 첫 범주 제외' if op.get('drop_first') else ''})"
    if k == "tukey":
        return f"{op['new']} = {op['source']}^{op['power']} (Tukey, 0은 log)"
    if k == "arith":
        return f"{op['new']} = {op['left']} {op['operator']} {op.get('right') or op.get('constant')}"
    if k == "transform":
        return f"{op['new']} = {op['kind']}({op['source']})"
    if k == "date":
        return f"{op['source']} 날짜 분해({', '.join(op.get('parts', []))})"
    if k == "groupagg":
        return f"{op['new']} = {op['group']}별 {op['source']} {op.get('agg', 'mean')}{' 편차' if op.get('deviation') else ''}"
    if k == "condition":
        return f"{op['new']} = ({op['source']} {op['operator']} {op['value']}) ? {op.get('true_val', 1)} : {op.get('false_val', 0)}"
    if k == "lump":
        return f"{op['new']} = {op['source']} 상위 {op.get('top_n')}범주 + 기타"
    if k == "expr":
        return f"{op['new']} = {op['expr']}"
    return str(op)


def new_columns(before: pd.DataFrame, after: pd.DataFrame) -> list[str]:
    return [c for c in after.columns if c not in before.columns] or ([] if before.shape == after.shape else [])


def tukey_ladder(df: pd.DataFrame, col: str) -> list:
    s = _num(df[col]).dropna()
    if s.empty:
        raise ValueError("수치값이 없습니다.")
    rows = []
    sample = s.sample(min(len(s), 5000), random_state=0)
    for p in [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]:
        if p == 0:
            v = np.log1p(sample.clip(lower=-0.999999))
            label = "log(x+1)"
        elif p < 0:
            v = np.power(sample.where(sample != 0) + 1e-9, p)
            label = f"x^{p}"
        else:
            v = np.power(sample.clip(lower=0) + 1e-9, p) if p != int(p) else np.power(sample, p)
            label = "x (원본)" if p == 1 else f"x^{p}"
        v = pd.Series(v).replace([np.inf, -np.inf], np.nan).dropna()
        if len(v) < 8 or v.nunique() < 3:
            continue
        sh = stats.shapiro(v.sample(min(len(v), 5000), random_state=0))
        rows.append({"거듭제곱": p, "변환": label, "왜도": float(v.skew()), "첨도": float(v.kurt()),
                     "Shapiro W": float(sh.statistic), "p-value": float(sh.pvalue)})
    tab = pd.DataFrame(rows)
    best = tab.iloc[(tab["왜도"].abs()).argmin()]
    from .charts import style
    import plotly.graph_objects as go
    fig = go.Figure(go.Bar(x=tab["변환"], y=tab["왜도"].abs(), marker_color=["#f97316" if r == best["변환"] else "#2563eb" for r in tab["변환"]]))
    style(fig, f"{col}: 거듭제곱별 |왜도|", "변환", "|왜도| (0에 가까울수록 대칭)", height=320, legend=False)
    return [B.table(tab, f"{col} Tukey 거듭제곱 사다리 비교"), B.chart(fig),
            B.interpret(f"왜도 절댓값이 가장 작은 변환은 **{best['변환']}** (거듭제곱 {best['거듭제곱']})입니다. "
                        "오른쪽으로 꼬리가 긴 분포는 0.5·0(log)·-1 쪽, 왼쪽으로 꼬리가 긴 분포는 2·3 쪽이 대칭에 가까워집니다. "
                        "선형 모델이나 잔차 정규성이 중요할 때 참고하고, 트리 모델은 단조 변환에 영향을 거의 받지 않습니다.")]
