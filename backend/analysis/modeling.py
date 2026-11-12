"""8~10단계: 데이터 분할, 모델링 전처리 파이프라인, 모델 학습·평가."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from sklearn import metrics as M
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit, KFold, RandomizedSearchCV, StratifiedKFold,
                                     TimeSeriesSplit, cross_validate, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler

from . import blocks as B
from .charts import PALETTE, style
from .recipe import LearnedOps

# ---------------------------------------------------------------- 분할


def split_data(df: pd.DataFrame, target: str, features: list[str], ttype: str, method: str = "random",
               test_size: float = 0.3, seed: int = 42, group_col: str | None = None, time_col: str | None = None,
               dropna_all: bool = False) -> tuple[np.ndarray, np.ndarray, dict]:
    data = df.dropna(subset=[target])
    dropped_target = len(df) - len(data)
    dropped_feat = 0
    if dropna_all:
        n0 = len(data)
        data = data.dropna(subset=features)
        dropped_feat = n0 - len(data)
    idx = data.index.values
    y = data[target]
    info = {"dropped_target": int(dropped_target), "dropped_features": int(dropped_feat), "method": method}
    if method == "stratified" and ttype != "regression":
        tr, te = train_test_split(idx, test_size=test_size, random_state=seed, stratify=y.astype(str))
    elif method == "stratified" and ttype == "regression":
        bins = pd.qcut(y, q=min(10, max(2, len(y) // 50)), duplicates="drop").astype(str)
        tr, te = train_test_split(idx, test_size=test_size, random_state=seed, stratify=bins)
    elif method == "group":
        if not group_col:
            raise ValueError("그룹 분할에는 그룹 열이 필요합니다.")
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        a, b = next(gss.split(idx, groups=data[group_col].astype(str)))
        tr, te = idx[a], idx[b]
    elif method == "time":
        if not time_col:
            raise ValueError("시간 분할에는 시간 열이 필요합니다.")
        order = data.sort_values(time_col).index.values
        cut = int(len(order) * (1 - test_size))
        tr, te = order[:cut], order[cut:]
    else:
        tr, te = train_test_split(idx, test_size=test_size, random_state=seed)
    return np.asarray(tr), np.asarray(te), info


def split_blocks(df, tr, te, target, ttype, features, info, group_col=None) -> list:
    train, test = df.loc[tr], df.loc[te]
    out = [B.metrics([("학습 데이터", len(train)), ("평가 데이터(홀드아웃)", len(test)), ("특징 수", len(features)),
                      ("타깃 결측 제외", info["dropped_target"]), ("결측 행 제외", info["dropped_features"]),
                      ("학습·평가 중복 행", len(set(tr) & set(te)))])]
    if ttype == "regression":
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=train[target], name="학습", opacity=0.6, histnorm="probability density", marker_color=PALETTE[0]))
        fig.add_trace(go.Histogram(x=test[target], name="평가", opacity=0.6, histnorm="probability density", marker_color=PALETTE[1]))
        fig.update_layout(barmode="overlay")
        style(fig, f"분할 후 {target} 분포 비교", target, "밀도", height=360)
        desc = pd.DataFrame({"학습": train[target].describe(), "평가": test[target].describe()}).T.reset_index().rename(columns={"index": "구분"})
        out += [B.chart(fig), B.table(desc, "학습·평가 타깃 기술통계")]
    else:
        a = train[target].astype(str).value_counts(normalize=True) * 100
        b = test[target].astype(str).value_counts(normalize=True) * 100
        t = pd.DataFrame({"학습(%)": a, "평가(%)": b}).fillna(0).round(2).reset_index().rename(columns={"index": "클래스"})
        fig = go.Figure([go.Bar(name="학습", x=t.iloc[:, 0], y=t["학습(%)"], marker_color=PALETTE[0]),
                         go.Bar(name="평가", x=t.iloc[:, 0], y=t["평가(%)"], marker_color=PALETTE[1])])
        style(fig, "분할 후 클래스 비율", target, "%", height=340)
        out += [B.chart(fig), B.table(t, "클래스 비율 비교")]
    rows = []
    for c in features:
        s1, s2 = train[c], test[c]
        if pd.api.types.is_numeric_dtype(s1):
            a, b = s1.dropna(), s2.dropna()
            if len(a) < 3 or len(b) < 3:
                continue
            ks = stats.ks_2samp(a, b)
            smd = (a.mean() - b.mean()) / (np.sqrt((a.var() + b.var()) / 2) or 1)
            rows.append({"변수": c, "학습 평균": a.mean(), "평가 평균": b.mean(), "표준화 평균차": smd, "KS p-value": ks.pvalue,
                         "판정": "치우침" if abs(smd) > 0.1 and ks.pvalue < 0.05 else "양호"})
    if rows:
        t = pd.DataFrame(rows).sort_values("표준화 평균차", key=lambda s: s.abs(), ascending=False)
        out.append(B.table(t, "학습·평가 특징 분포 비교"))
        bad = t[t["판정"] == "치우침"]["변수"].tolist()
        out.append(B.interpret(
            ("분할 후 분포가 치우친 변수: " + ", ".join(bad) + ". 시드를 바꾸거나 층화 분할을 검토하세요. " if bad else
             "학습과 평가 데이터의 특징 분포가 비슷해 분할이 고르게 되었습니다. ")
            + "표준화 평균차 |d|>0.1이면서 KS 검정이 유의하면 치우침으로 표시합니다. "
            + {"time": "시간 분할은 미래 데이터로 평가하므로 분포 차이가 자연스럽게 생길 수 있고, 실제 운영 성능에 더 가깝습니다. ",
               "group": f"그룹 분할은 같은 `{group_col}`이 학습과 평가에 동시에 들어가지 않도록 해, 새 그룹(로트·설비)에 대한 일반화를 평가합니다. "}.get(info["method"], "")
            + "평가 데이터는 10단계 최종 평가 전까지 모델 선택에 쓰지 않습니다."))
    return out


# ---------------------------------------------------------------- 전처리


class Caster(BaseEstimator, TransformerMixin):
    """지정 열을 문자열 범주로 바꾸고, 학습 시 열 목록을 고정한다."""

    def __init__(self, as_categorical=None):
        self.as_categorical = as_categorical

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns_ = list(X.columns)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for c in self.columns_:
            if c not in X.columns:
                X[c] = np.nan
        X = X[self.columns_]
        for c in (self.as_categorical or []):
            if c in X.columns:
                X[c] = X[c].map(lambda v: v if pd.isna(v) else str(v)).astype(object)
        for c in X.columns:
            if pd.api.types.is_bool_dtype(X[c]):
                X[c] = X[c].astype(int)
            elif not pd.api.types.is_numeric_dtype(X[c]):
                X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns_, dtype=object)


class TopCategories(BaseEstimator, TransformerMixin):
    """학습 데이터 빈도 상위 N개 범주만 남기고 나머지는 '기타'로."""

    def __init__(self, top_n=5):
        self.top_n = top_n

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.keep_ = {c: X[c].astype(str).value_counts().head(self.top_n).index.tolist() for c in X.columns}
        self.columns_ = list(X.columns)
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columns_).copy()
        for c in self.columns_:
            s = X[c].astype(str)
            X[c] = s.where(s.isin(self.keep_[c]), "기타")
            X.loc[pd.DataFrame(X)[c].isna(), c] = "결측"
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns_, dtype=object)


def select_numeric(X):
    return [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]


def select_categorical(X):
    return [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]


SCALERS = {"standard": StandardScaler, "robust": RobustScaler, "minmax": MinMaxScaler}


def build_preprocessor(cfg: dict, learned_ops: list[dict]) -> Pipeline:
    scaler = cfg.get("scaler", "standard")
    num_steps = [("impute", SimpleImputer(strategy=cfg.get("num_impute", "median")))]
    if scaler in SCALERS:
        num_steps.append(("scale", SCALERS[scaler]()))
    enc = cfg.get("encoding", "onehot")
    if enc == "target":
        from sklearn.preprocessing import TargetEncoder
        cat_steps = [("top", TopCategories(int(cfg.get("top_n", 5)))), ("encode", TargetEncoder(random_state=0))]
    else:
        cat_steps = [("top", TopCategories(int(cfg.get("top_n", 5)))),
                     ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="if_binary" if cfg.get("drop_binary") else None))]
    ct = ColumnTransformer([("num", Pipeline(num_steps), select_numeric), ("cat", Pipeline(cat_steps), select_categorical)],
                           verbose_feature_names_out=False)
    steps = []
    if learned_ops:
        steps.append(("learned", LearnedOps(learned_ops)))
    steps += [("cast", Caster(cfg.get("as_categorical") or [])), ("columns", ct)]
    pipe = Pipeline(steps)
    pipe.set_output(transform="pandas")
    return pipe


def preprocess_blocks(pre: Pipeline, Xtr: pd.DataFrame, Xte: pd.DataFrame, cfg: dict, learned: list) -> list:
    t0 = time.time()
    Ptr = pre.fit_transform(Xtr)
    Pte = pre.transform(Xte)
    ct = pre.named_steps["columns"]
    num = ct.transformers_[0][2] if ct.transformers_ else []
    cat = ct.transformers_[1][2] if len(ct.transformers_) > 1 else []
    rows = []
    for c in Ptr.columns:
        rows.append({"전처리 후 특징": c, "학습 평균": float(Ptr[c].mean()), "학습 표준편차": float(Ptr[c].std()),
                     "평가 평균": float(Pte[c].mean()) if c in Pte else None})
    out = [B.metrics([("원 특징", Xtr.shape[1]), ("연속형", len(num)), ("범주형", len(cat)), ("전처리 후 특징", Ptr.shape[1]),
                      ("처리 시간(초)", round(time.time() - t0, 2))])]
    steps = []
    if learned:
        from .recipe import describe_learned
        steps += [f"학습형 처리(학습 데이터로 적합): {describe_learned(o)}" for o in learned]
    steps.append(f"연속형: 잔여 결측 {cfg.get('num_impute', 'median')} 대체 → {({'standard': '표준화(평균0·표준편차1)', 'robust': '강건 스케일(중앙값·IQR)', 'minmax': '0~1 정규화'}).get(cfg.get('scaler', 'standard'), '스케일 없음')}")
    steps.append(f"범주형: 학습 데이터 빈도 상위 {cfg.get('top_n', 5)}개 + 기타 → {'타깃 인코딩(교차 적합)' if cfg.get('encoding') == 'target' else '원-핫 인코딩'}")
    out.append(B.text("**전처리 파이프라인**\n" + "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))))
    out.append(B.table(Ptr.head(10), "전처리된 학습 데이터 (상위 10행)"))
    out.append(B.table(Pte.head(10), "전처리된 평가 데이터 (상위 10행)"))
    out.append(B.table(pd.DataFrame(rows), "특징별 전처리 결과 요약"))
    out.append(B.interpret(
        "모든 기준값(평균·표준편차·상위 범주·대체값)은 학습 데이터에서만 계산하고 평가 데이터에는 그 값을 그대로 적용합니다. "
        "그래서 평가 데이터의 평균이 정확히 0이 아니어도 정상입니다. 평가 데이터에서 다시 계산하면 평가 정보가 학습에 새어 들어가 성능이 부풀려집니다."))
    return out


# ---------------------------------------------------------------- 모델 카탈로그


def catalog(ttype: str, n_classes: int = 2) -> dict:
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier,
                                  GradientBoostingRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                                  RandomForestClassifier, RandomForestRegressor)
    from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.svm import SVC, SVR
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    cat = {}
    if ttype == "regression":
        cat = {
            "baseline": ("기준선(평균)", DummyRegressor(), {}, "기준"),
            "linear": ("회귀분석", LinearRegression(), {}, "기준"),
            "tree": ("의사결정나무", DecisionTreeRegressor(random_state=42, max_depth=5),
                     {"max_depth": [2, 3, 4, 5, 6, 8, 10, None], "min_samples_leaf": [1, 2, 5, 10, 20]}, "기준"),
            "rf": ("랜덤포레스트", RandomForestRegressor(random_state=42, n_estimators=200),
                   {"n_estimators": [100, 200, 400], "max_depth": [None, 5, 10, 20], "min_samples_leaf": [1, 2, 5], "max_features": [1.0, "sqrt", 0.5]}, "기준"),
            "mlp": ("인공신경망", MLPRegressor(random_state=42, max_iter=1000, early_stopping=True),
                    {"hidden_layer_sizes": [(50,), (100,), (100, 50), (64, 32)], "alpha": [1e-4, 1e-3, 1e-2, 1e-1]}, "기준"),
            "ridge": ("릿지(Ridge)", Ridge(alpha=1.0), {"alpha": list(np.logspace(-3, 3, 13))}, "기준"),
            "lasso": ("라쏘(Lasso)", Lasso(alpha=1.0, max_iter=10000), {"alpha": list(np.logspace(-4, 1, 11))}, "기준"),
            "enet": ("엘라스틱넷", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000),
                     {"alpha": list(np.logspace(-4, 1, 11)), "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]}, "심화"),
            "knn": ("KNN", KNeighborsRegressor(), {"n_neighbors": [3, 5, 7, 11, 15, 25], "weights": ["uniform", "distance"]}, "심화"),
            "svm": ("SVM(RBF)", SVR(), {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.01, 0.1], "epsilon": [0.01, 0.1, 0.5]}, "심화"),
            "gbm": ("그래디언트부스팅", GradientBoostingRegressor(random_state=42),
                    {"n_estimators": [100, 200, 400], "learning_rate": [0.03, 0.1, 0.2], "max_depth": [2, 3, 4, 5], "subsample": [0.7, 1.0]}, "심화"),
            "hgb": ("히스토그램 부스팅", HistGradientBoostingRegressor(random_state=42),
                    {"learning_rate": [0.03, 0.1, 0.2], "max_leaf_nodes": [15, 31, 63], "l2_regularization": [0, 0.1, 1.0]}, "심화"),
            "et": ("엑스트라트리", ExtraTreesRegressor(random_state=42, n_estimators=300),
                   {"n_estimators": [200, 400], "max_depth": [None, 10, 20], "min_samples_leaf": [1, 2, 5]}, "심화"),
        }
        try:
            from xgboost import XGBRegressor
            cat["xgb"] = ("XGB", XGBRegressor(random_state=42, objective="reg:squarederror", n_estimators=300, verbosity=0, n_jobs=1),
                          {"n_estimators": [200, 400, 800], "learning_rate": [0.03, 0.1, 0.2], "max_depth": [3, 4, 6, 8], "subsample": [0.7, 1.0], "colsample_bytree": [0.7, 1.0]}, "기준")
        except Exception:
            pass
        try:
            from lightgbm import LGBMRegressor
            cat["lgbm"] = ("LightGBM", LGBMRegressor(random_state=42, verbose=-1, n_jobs=1),
                           {"n_estimators": [200, 400, 800], "learning_rate": [0.03, 0.1], "num_leaves": [15, 31, 63], "subsample": [0.7, 1.0]}, "심화")
        except Exception:
            pass
    else:
        cat = {
            "baseline": ("기준선(최빈값)", DummyClassifier(strategy="most_frequent"), {}, "기준"),
            "logit": ("로지스틱회귀분석", LogisticRegression(max_iter=2000), {"C": list(np.logspace(-3, 3, 13))}, "기준"),
            "tree": ("의사결정나무", DecisionTreeClassifier(random_state=42, max_depth=5),
                     {"max_depth": [2, 3, 4, 5, 6, 8, 10, None], "min_samples_leaf": [1, 2, 5, 10, 20]}, "기준"),
            "rf": ("랜덤포레스트", RandomForestClassifier(random_state=42, n_estimators=200),
                   {"n_estimators": [100, 200, 400], "max_depth": [None, 5, 10, 20], "min_samples_leaf": [1, 2, 5], "max_features": ["sqrt", 0.5, 1.0]}, "기준"),
            "mlp": ("인공신경망", MLPClassifier(random_state=42, max_iter=1000, early_stopping=True),
                    {"hidden_layer_sizes": [(50,), (100,), (100, 50), (64, 32)], "alpha": [1e-4, 1e-3, 1e-2, 1e-1]}, "기준"),
            "knn": ("KNN", KNeighborsClassifier(), {"n_neighbors": [3, 5, 7, 11, 15, 25], "weights": ["uniform", "distance"]}, "심화"),
            "svm": ("SVM(RBF)", SVC(probability=True, random_state=42), {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.01, 0.1]}, "심화"),
            "gbm": ("그래디언트부스팅", GradientBoostingClassifier(random_state=42),
                    {"n_estimators": [100, 200, 400], "learning_rate": [0.03, 0.1, 0.2], "max_depth": [2, 3, 4], "subsample": [0.7, 1.0]}, "심화"),
            "hgb": ("히스토그램 부스팅", HistGradientBoostingClassifier(random_state=42),
                    {"learning_rate": [0.03, 0.1, 0.2], "max_leaf_nodes": [15, 31, 63], "l2_regularization": [0, 0.1, 1.0]}, "심화"),
            "et": ("엑스트라트리", ExtraTreesClassifier(random_state=42, n_estimators=300),
                   {"n_estimators": [200, 400], "max_depth": [None, 10, 20], "min_samples_leaf": [1, 2, 5]}, "심화"),
        }
        try:
            from xgboost import XGBClassifier
            cat["xgb"] = ("XGB", XGBClassifier(random_state=42, eval_metric="logloss", n_estimators=300, verbosity=0, n_jobs=1),
                          {"n_estimators": [200, 400, 800], "learning_rate": [0.03, 0.1, 0.2], "max_depth": [3, 4, 6, 8], "subsample": [0.7, 1.0], "colsample_bytree": [0.7, 1.0]}, "기준")
        except Exception:
            pass
        try:
            from lightgbm import LGBMClassifier
            cat["lgbm"] = ("LightGBM", LGBMClassifier(random_state=42, verbose=-1, n_jobs=1),
                           {"n_estimators": [200, 400, 800], "learning_rate": [0.03, 0.1], "num_leaves": [15, 31, 63], "subsample": [0.7, 1.0]}, "심화")
        except Exception:
            pass
    return cat


def model_list(ttype: str) -> list[dict]:
    return [{"key": k, "name": v[0], "tier": v[3], "tunable": bool(v[2])} for k, v in catalog(ttype).items()]


# ---------------------------------------------------------------- 지표

REG_SCORING = {"R2_Score": "r2", "RMSE": "neg_root_mean_squared_error", "MAE": "neg_mean_absolute_error",
               "MAPE": "neg_mean_absolute_percentage_error"}


def reg_metrics(y, p, n_features=None) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    r2 = M.r2_score(y, p)
    out = {"R2_Score": r2, "RMSE": float(np.sqrt(M.mean_squared_error(y, p))), "MAE": M.mean_absolute_error(y, p),
           "MAPE": M.mean_absolute_percentage_error(y, p) * 100}
    if n_features and len(y) - n_features - 1 > 0:
        out["Adj_R2"] = 1 - (1 - r2) * (len(y) - 1) / (len(y) - n_features - 1)
    return out


def clf_metrics(y, pred, proba, classes: list) -> dict:
    y = np.asarray(y)
    k = len(classes)
    out = {"Accuracy": M.accuracy_score(y, pred), "Balanced_Acc": M.balanced_accuracy_score(y, pred),
           "MCC": M.matthews_corrcoef(y, pred)}
    if k == 2:
        for lab in (0, 1):
            out[f"Recall_{classes[lab]}"] = M.recall_score(y, pred, pos_label=lab, zero_division=0)
            out[f"Precision_{classes[lab]}"] = M.precision_score(y, pred, pos_label=lab, zero_division=0)
            out[f"F1_{classes[lab]}"] = M.f1_score(y, pred, pos_label=lab, zero_division=0)
        if proba is not None and len(np.unique(y)) == 2:
            out["AUC"] = M.roc_auc_score(y, proba[:, 1])
            out["PR_AUC"] = M.average_precision_score(y, proba[:, 1])
            out["LogLoss"] = M.log_loss(y, np.clip(proba, 1e-12, 1), labels=[0, 1])
    else:
        out["F1_macro"] = M.f1_score(y, pred, average="macro", zero_division=0)
        out["F1_weighted"] = M.f1_score(y, pred, average="weighted", zero_division=0)
        out["Recall_macro"] = M.recall_score(y, pred, average="macro", zero_division=0)
        out["Precision_macro"] = M.precision_score(y, pred, average="macro", zero_division=0)
        if proba is not None and len(np.unique(y)) == k:
            try:
                out["AUC_ovr"] = M.roc_auc_score(y, proba, multi_class="ovr")
            except ValueError:
                pass
            out["LogLoss"] = M.log_loss(y, np.clip(proba, 1e-12, 1), labels=list(range(k)))
    return out


def cv_splitter(method: str, ttype: str, folds: int, seed: int):
    if method == "time":
        return TimeSeriesSplit(n_splits=folds)
    if method == "group":
        return GroupKFold(n_splits=folds)
    if ttype != "regression":
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return KFold(n_splits=folds, shuffle=True, random_state=seed)


def scoring_for(ttype: str, k: int) -> dict:
    if ttype == "regression":
        return dict(REG_SCORING)
    if k == 2:
        return {"Accuracy": "accuracy", "F1": "f1", "AUC": "roc_auc", "Balanced_Acc": "balanced_accuracy"}
    return {"Accuracy": "accuracy", "F1_macro": "f1_macro", "Balanced_Acc": "balanced_accuracy"}


SELECT_METRICS = {
    "regression": {"RMSE": False, "R2_Score": True, "MAE": False, "MAPE": False},
    "binary": {"AUC": True, "F1": True, "Accuracy": True, "Balanced_Acc": True},
    "multiclass": {"F1_macro": True, "Accuracy": True, "Balanced_Acc": True},
}


def predict_with(pipe, X, ttype):
    pred = pipe.predict(X)
    proba = None
    if ttype != "regression" and hasattr(pipe, "predict_proba"):
        try:
            proba = pipe.predict_proba(X)
        except Exception:
            proba = None
    return pred, proba


def train_models(pre: Pipeline, keys: list[str], Xtr, ytr, Xte, yte, ttype: str, classes: list, split_method: str,
                 groups=None, folds: int = 5, seed: int = 42, tune: bool = False, n_iter: int = 15,
                 select_metric: str | None = None, class_weight: bool = False,
                 progress=None, cancelled=None) -> dict:
    cat = catalog(ttype)
    k = len(classes) if classes else 0
    scoring = scoring_for(ttype, k)
    cv = cv_splitter(split_method, ttype, folds, seed)
    results = {}
    for i, key in enumerate(keys):
        if cancelled and cancelled():
            break
        name, est, space, tier = cat[key]
        est = clone(est)
        try:
            est.set_params(random_state=seed)
        except ValueError:
            pass
        if class_weight and ttype != "regression":
            try:
                est.set_params(class_weight="balanced")
            except ValueError:
                pass
        pipe = Pipeline([("pre", clone(pre)), ("model", est)])
        if progress:
            progress(i / max(len(keys), 1), f"{name} 학습 중 ({i + 1}/{len(keys)})")
        t0 = time.time()
        best_params = None
        fit_kw = {}
        if tune and space:
            n_combo = int(np.prod([len(v) for v in space.values()]))
            search = RandomizedSearchCV(pipe, {f"model__{p}": v for p, v in space.items()}, n_iter=min(n_iter, n_combo),
                                        scoring=list(scoring.values())[0] if ttype == "regression" else ("roc_auc" if k == 2 else "f1_macro"),
                                        cv=cv, random_state=seed, n_jobs=-1, error_score=np.nan)
            search.fit(Xtr, ytr, **({"groups": groups} if groups is not None else {}))
            pipe = search.best_estimator_
            best_params = {p.replace("model__", ""): v for p, v in search.best_params_.items()}
        try:
            cvres = cross_validate(clone(pipe), Xtr, ytr, cv=cv, scoring=scoring, groups=groups, n_jobs=-1, error_score=np.nan)
        except Exception as e:  # noqa
            cvres = {}
        cv_summary = {}
        for m, sk in scoring.items():
            arr = cvres.get(f"test_{m}")
            if arr is None:
                continue
            arr = np.asarray(arr, dtype=float)
            if sk.startswith("neg_"):
                arr = -arr
            if m == "MAPE":
                arr = arr * 100
            cv_summary[m] = {"mean": float(np.nanmean(arr)), "std": float(np.nanstd(arr)), "folds": arr.tolist()}
        pipe.fit(Xtr, ytr)
        fit_time = time.time() - t0
        ptr, prtr = predict_with(pipe, Xtr, ttype)
        pte, prte = predict_with(pipe, Xte, ttype)
        nfeat = None
        try:
            nfeat = len(pipe.named_steps["pre"].get_feature_names_out())
        except Exception:
            pass
        if ttype == "regression":
            mtr, mte = reg_metrics(ytr, ptr, nfeat), reg_metrics(yte, pte, nfeat)
        else:
            mtr, mte = clf_metrics(ytr, ptr, prtr, classes), clf_metrics(yte, pte, prte, classes)
        results[key] = {"key": key, "name": name, "tier": tier, "pipe": pipe, "train": mtr, "test": mte, "cv": cv_summary,
                        "fit_time": fit_time, "best_params": best_params,
                        "test_pred": pte, "test_proba": prte, "train_pred": ptr}
    # 최적 모델 선정
    ttkey = ttype if ttype == "regression" else ("binary" if k == 2 else "multiclass")
    sel = select_metric or list(SELECT_METRICS[ttkey].keys())[0]
    higher = SELECT_METRICS[ttkey].get(sel, True)
    cands = [r for r in results.values() if r["key"] != "baseline"] or list(results.values())

    def score(r):
        if sel in r["cv"]:
            v = r["cv"][sel]["mean"]
        else:
            v = r["test"].get(sel, r["test"].get("AUC", r["test"].get("Accuracy")))
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return -np.inf
        return v if higher else -v

    best = max(cands, key=score)["key"] if cands else None
    return {"results": results, "best": best, "select_metric": sel, "folds": folds,
            "selected_by": "교차검증"}


def training_blocks(out: dict, ttype: str, classes: list) -> list:
    res = out["results"]
    best = out["best"]
    sel = out["select_metric"]
    tr_rows, te_rows, cv_rows = [], [], []
    for r in res.values():
        tr_rows.append({"모델명": r["name"], **r["train"]})
        te_rows.append({"모델명": r["name"], **r["test"], "학습시간(초)": round(r["fit_time"], 2)})
        cv_rows.append({"모델명": r["name"], **{f"{m} 평균": v["mean"] for m, v in r["cv"].items()},
                        **{f"{m} 표준편차": v["std"] for m, v in r["cv"].items()}})
    trd, ted, cvd = pd.DataFrame(tr_rows), pd.DataFrame(te_rows), pd.DataFrame(cv_rows)
    bname = res[best]["name"] if best else "-"
    main = "RMSE" if ttype == "regression" else ("AUC" if len(classes) == 2 else "F1_macro")
    over = []
    for r in res.values():
        if ttype == "regression":
            a, b = r["train"]["R2_Score"], r["test"]["R2_Score"]
            c = r["cv"].get("R2_Score", {}).get("mean")
            over.append({"모델명": r["name"], "학습 R²": a, "교차검증 R²": c, "평가 R²": b, "학습-평가 차이": a - b,
                         "진단": "과적합 의심" if a - b > 0.1 else ("과소적합 의심" if b < 0.3 else "양호")})
        else:
            a, b = r["train"]["Accuracy"], r["test"]["Accuracy"]
            c = r["cv"].get("Accuracy", {}).get("mean")
            over.append({"모델명": r["name"], "학습 정확도": a, "교차검증 정확도": c, "평가 정확도": b, "학습-평가 차이": a - b,
                         "진단": "과적합 의심" if a - b > 0.05 else "양호"})
    base = res.get("baseline")
    bm = res[best]["test"] if best else {}
    lift = ""
    if base and best:
        if ttype == "regression":
            lift = f"기준선(평균 예측) RMSE {base['test']['RMSE']:.4g} 대비 {bname} RMSE {bm['RMSE']:.4g}로 오차가 {(1 - bm['RMSE'] / base['test']['RMSE']) * 100:.1f}% 줄었습니다. "
        else:
            lift = f"기준선(최빈값) 정확도 {base['test']['Accuracy']:.3f} 대비 {bname} 정확도 {bm['Accuracy']:.3f}입니다. "
    params = [{"모델명": r["name"], "최적 하이퍼파라미터": str(r["best_params"])} for r in res.values() if r["best_params"]]
    blocks = [
        B.note(f"최적 모델: **{bname}** · 선정 기준: {out['selected_by']} {sel}", "success"),
        B.metrics([(k, v) for k, v in list(bm.items())[:6]], f"{bname} 평가 데이터 성능"),
        B.table(trd, "학습용 성능평가", download="train_metrics.csv"),
        B.table(ted, "평가용 성능평가 (홀드아웃)", download="test_metrics.csv"),
        B.table(cvd, f"교차검증 성능 (학습 데이터 내부 {out.get('folds', 5)}겹, 평균±표준편차)"),
        B.table(pd.DataFrame(over), "과적합 진단"),
    ]
    if params:
        blocks.append(B.table(pd.DataFrame(params), "하이퍼파라미터 튜닝 결과"))
    blocks.append(B.interpret(
        lift + ("회귀 지표: R²는 설명된 변동 비율(1에 가까울수록 좋음), RMSE·MAE는 타깃과 같은 단위의 평균 오차(작을수록 좋음), MAPE는 상대 오차(%)입니다. "
                if ttype == "regression" else
                "분류 지표: 정확도는 전체 맞힌 비율, 재현율은 실제 해당 클래스를 놓치지 않은 비율, 정밀도는 그렇다고 예측한 것 중 맞은 비율, "
                "AUC는 임계값과 무관한 순위 성능(0.5=무작위, 1=완벽), MCC는 불균형에도 공정한 종합 지표입니다. ")
        + "최적 모델은 평가 데이터가 아니라 학습 데이터 내부 교차검증으로 골랐으므로, 평가용 성능은 처음 보는 데이터에 대한 정직한 추정입니다. "
        "학습 성능이 평가 성능보다 크게 높으면 과적합이므로 더 단순한 모델이나 규제를 고려하세요."))
    return blocks
