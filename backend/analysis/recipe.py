"""레시피: 결정적 연산(파생변수·행 삭제)과 학습형 연산(결측 대체·이상치 처리)을 관리한다.

- 결정적 연산은 결합 데이터에서 순서대로 다시 계산해 작업 데이터를 만든다.
- 학습형 연산은 sklearn 변환기(LearnedOps)로 구현해,
  개선 방식에서는 분할 후 학습 데이터에만 적합하고, 기준 스크립트 방식에서는 분할 전 전체에 적합한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from . import features

ROW_OPS = {"drop_missing", "drop_outliers", "dedupe", "filter"}


def apply_row_op(df: pd.DataFrame, op: dict) -> pd.DataFrame:
    k = op["op"]
    if k == "drop_missing":
        cols = [c for c in op["columns"] if c in df.columns]
        return df.dropna(subset=cols)
    if k == "drop_outliers":
        c = op["column"]
        s = pd.to_numeric(df[c], errors="coerce")
        keep = s.isna() | ((s >= op["lower"]) & (s <= op["upper"]))
        return df[keep]
    if k == "dedupe":
        return df.drop_duplicates(subset=op.get("columns") or None)
    if k == "filter":
        return df.query(op["query"], engine="python")
    raise ValueError(k)


def build_working(merged: pd.DataFrame, recipe: list[dict]) -> tuple[pd.DataFrame, list[dict]]:
    """레시피를 순서대로 적용하고 단계별 행·열 수를 기록한다."""
    df = merged
    trace = []
    for op in recipe:
        before = df.shape
        if op["op"] in ROW_OPS:
            df = apply_row_op(df, op)
        else:
            df = features.apply_op(df, op)
        trace.append({"연산": describe(op), "전": f"{before[0]:,}×{before[1]}", "후": f"{df.shape[0]:,}×{df.shape[1]}",
                      "행 변화": df.shape[0] - before[0]})
    return df.reset_index(drop=True), trace


def apply_for_new_data(df: pd.DataFrame, recipe: list[dict]) -> tuple[pd.DataFrame, list[str]]:
    """신규 데이터에는 파생변수만 재적용한다(행 삭제는 하지 않음)."""
    return features.apply_all(df, [op for op in recipe if op["op"] not in ROW_OPS])


def describe(op: dict) -> str:
    k = op["op"]
    if k == "drop_missing":
        return f"결측 행 삭제: {', '.join(op['columns'])}"
    if k == "drop_outliers":
        return f"이상치 행 제거: {op['column']} ∉ [{op['lower']:.4g}, {op['upper']:.4g}]"
    if k == "dedupe":
        return "중복 행 제거" + (f" ({', '.join(op['columns'])})" if op.get("columns") else "")
    if k == "filter":
        return f"행 필터: {op['query']}"
    if k in ("impute", "indicator", "clip"):
        return describe_learned(op)
    return features.describe(op)


def describe_learned(op: dict) -> str:
    k = op["op"]
    if k == "impute":
        m = {"mean": "평균", "median": "중앙값", "mode": "최빈값", "constant": f"상수({op.get('value')})",
             "knn": f"KNN(k={op.get('k', 5)})", "iterative": "반복 대체(회귀)", "ffill": "직전 값"}[op["method"]]
        return f"결측 대체[{m}]: {', '.join(op['columns'])}"
    if k == "indicator":
        return f"결측 지시변수 추가: {', '.join(op['columns'])}"
    if k == "clip":
        m = {"iqr": f"IQR×{op.get('factor', 1.5)}", "zscore": f"Z>{op.get('factor', 3)}", "mad": f"수정Z>{op.get('factor', 3.5)}",
             "percentile": f"백분위 {op.get('lower_pct', 1)}–{op.get('upper_pct', 99)}%", "custom": f"[{op.get('lower')}, {op.get('upper')}]"}[op["method"]]
        how = "정상범위 최소/최대로 대체" if op.get("mode") == "normal_minmax" else ("결측 처리" if op.get("mode") == "nan" else "경계값으로 대체(윈저화)")
        return f"이상치 {how}[{m}]: {', '.join(op['columns'])}"
    return str(op)


def bounds(s: pd.Series, method: str, factor: float | None = None, lower=None, upper=None,
           lower_pct: float = 1, upper_pct: float = 99) -> tuple[float, float]:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return -np.inf, np.inf
    if method == "iqr":
        f = 1.5 if factor is None else factor
        q1, q3 = s.quantile([0.25, 0.75])
        return float(q1 - f * (q3 - q1)), float(q3 + f * (q3 - q1))
    if method == "zscore":
        f = 3 if factor is None else factor
        return float(s.mean() - f * s.std()), float(s.mean() + f * s.std())
    if method == "mad":
        f = 3.5 if factor is None else factor
        med = s.median()
        mad = (s - med).abs().median() * 1.4826
        mad = mad or s.std() or 1
        return float(med - f * mad), float(med + f * mad)
    if method == "percentile":
        return float(s.quantile(lower_pct / 100)), float(s.quantile(upper_pct / 100))
    if method == "custom":
        return float(lower if lower is not None else -np.inf), float(upper if upper is not None else np.inf)
    raise ValueError(method)


class LearnedOps(BaseEstimator, TransformerMixin):
    """학습형 전처리 연산 목록을 순서대로 적합·적용하는 DataFrame 변환기."""

    def __init__(self, ops=None):
        self.ops = ops

    def fit(self, X, y=None):
        self._fit_transform(pd.DataFrame(X).copy(), fit=True)
        return self

    def transform(self, X):
        return self._fit_transform(pd.DataFrame(X).copy(), fit=False)

    def fit_transform(self, X, y=None, **kw):
        return self._fit_transform(pd.DataFrame(X).copy(), fit=True)

    def _fit_transform(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        if fit:
            self.params_ = []
        for i, op in enumerate(self.ops or []):
            cols = [c for c in op.get("columns", []) if c in df.columns]
            k = op["op"]
            if fit:
                p = {}
            else:
                p = self.params_[i]
            if k == "indicator":
                for c in op.get("columns", []):
                    df[f"{c}_결측"] = (df[c].isna().astype(int) if c in df.columns else 1)
            elif k == "impute":
                m = op["method"]
                if m in ("mean", "median", "mode", "constant"):
                    for c in cols:
                        if fit:
                            s = df[c]
                            if m == "constant":
                                v = op.get("value", 0)
                            elif m == "mode" or not pd.api.types.is_numeric_dtype(s):
                                md = s.mode(dropna=True)
                                v = md.iloc[0] if len(md) else None
                            elif m == "mean":
                                v = float(s.mean())
                            else:
                                v = float(s.median())
                            p[c] = v
                        if p.get(c) is not None:
                            df[c] = df[c].fillna(p[c])
                elif m == "ffill":
                    for c in cols:
                        df[c] = df[c].ffill().bfill()
                elif m in ("knn", "iterative"):
                    num_ctx = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                    tcols = [c for c in cols if c in num_ctx]
                    if not tcols:
                        pass
                    else:
                        if fit:
                            if m == "knn":
                                from sklearn.impute import KNNImputer
                                imp = KNNImputer(n_neighbors=int(op.get("k", 5)))
                            else:
                                from sklearn.experimental import enable_iterative_imputer  # noqa: F401
                                from sklearn.impute import IterativeImputer
                                imp = IterativeImputer(random_state=0, max_iter=10)
                            ctx = df[num_ctx]
                            keep = [c for c in num_ctx if ctx[c].notna().any()]
                            imp.fit(ctx[keep])
                            p["imputer"], p["ctx"] = imp, keep
                        imp, keep = p["imputer"], p["ctx"]
                        ctx = df.reindex(columns=keep)
                        filled = pd.DataFrame(imp.transform(ctx), columns=keep, index=df.index)
                        for c in tcols:
                            if c in filled:
                                df[c] = df[c].fillna(filled[c])
            elif k == "clip":
                for c in cols:
                    s = pd.to_numeric(df[c], errors="coerce")
                    if fit:
                        lo, hi = bounds(s, op["method"], op.get("factor"), op.get("lower"), op.get("upper"),
                                        op.get("lower_pct", 1), op.get("upper_pct", 99))
                        normal = s[(s >= lo) & (s <= hi)]
                        p[c] = {"lo": lo, "hi": hi,
                                "nmin": float(normal.min()) if len(normal) else lo,
                                "nmax": float(normal.max()) if len(normal) else hi}
                    b = p[c]
                    mode = op.get("mode", "winsor")
                    if mode == "nan":
                        df[c] = s.where((s >= b["lo"]) & (s <= b["hi"]) | s.isna())
                    elif mode == "normal_minmax":
                        s2 = s.copy()
                        s2[s < b["lo"]] = b["nmin"]
                        s2[s > b["hi"]] = b["nmax"]
                        df[c] = s2
                    else:
                        df[c] = s.clip(b["lo"], b["hi"])
            if fit:
                self.params_.append(p)
        return df

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object) if input_features is not None else None


def view(working: pd.DataFrame, learned: list[dict]) -> pd.DataFrame:
    """탐색용 화면: 학습형 연산을 전체 데이터 기준으로 미리 적용한 데이터."""
    if not learned:
        return working
    return LearnedOps(learned).fit_transform(working)
