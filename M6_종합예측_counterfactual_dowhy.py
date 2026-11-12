# app.py
import io
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_percentage_error,
    accuracy_score, recall_score, precision_score, f1_score,
    roc_auc_score, roc_curve
)

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.linear_model import Ridge, Lasso
from sklearn.base import clone


try:
    from xgboost import XGBRegressor, XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False


st.set_page_config(page_title="데이터 분석 자동화 웹", layout="wide")

st.markdown(
    """
    <style>
    body {font-family: Malgun Gothic, sans-serif;}
    .stDataFrame {font-family: Malgun Gothic, sans-serif;}
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 공통 함수
# ============================================================
def read_uploaded_file(file):
    filename = (file.name or "").lower()

    if filename.endswith(".xlsx"):
        file.seek(0)
        return pd.read_excel(file, engine="openpyxl"), "xlsx"

    if filename.endswith(".xls"):
        file.seek(0)
        return pd.read_excel(file, engine="xlrd"), "xls"

    for enc in ["cp949", "utf-8", "euc-kr"]:
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=enc), enc
        except Exception:
            file.seek(0)
    raise ValueError("파일 인코딩을 확인할 수 없습니다.")


def download_csv_button(df, label, filename):
    csv = df.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        label=label,
        data=csv,
        file_name=filename,
        mime="text/csv"
    )


def download_excel_button(df, label, filename):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    st.download_button(
        label=label,
        data=buffer.getvalue(),
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


def get_numeric_cols(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()


def get_categorical_cols(df):
    return df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()


def create_binning_feature(df, source_col, new_col, n_bins=3, method="등간격"):
    if source_col not in df.columns:
        raise ValueError(f"컬럼 {source_col}이 없습니다.")

    s = pd.to_numeric(df[source_col], errors="coerce")
    if s.dropna().empty:
        raise ValueError("구간 분할에 사용할 수 있는 수치값이 없습니다.")

    if method == "분위수":
        bins = pd.qcut(s, q=n_bins, duplicates="drop")
    else:
        bins = pd.cut(s, bins=n_bins, duplicates="drop")

    if bins.isna().all():
        raise ValueError("구간 분할이 불가능합니다.")

    df[new_col] = bins.astype(str)
    return df


def create_dummy_feature(df, source_col, new_col_prefix, drop_first=False):
    if source_col not in df.columns:
        raise ValueError(f"컬럼 {source_col}이 없습니다.")

    dummies = pd.get_dummies(df[source_col], prefix=new_col_prefix, drop_first=drop_first, dtype=int)
    df = pd.concat([df.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    return df


def create_tukey_feature(df, source_col, new_col, power=1):
    if source_col not in df.columns:
        raise ValueError(f"컬럼 {source_col}이 없습니다.")

    s = pd.to_numeric(df[source_col], errors="coerce")
    if s.dropna().empty:
        raise ValueError("변환에 사용할 수 있는 수치값이 없습니다.")

    if power == 0:
        values = np.log1p(s)
    elif power == 1:
        values = s
    elif power == -1:
        values = 1 / (s + 1e-9)
    else:
        values = np.power(s + 1e-9, power)

    df[new_col] = values
    return df


def create_arithmetic_feature(df, left_col, right_col, operator, new_col):
    if left_col not in df.columns or right_col not in df.columns:
        raise ValueError("선택한 두 컬럼이 모두 존재해야 합니다.")

    left = pd.to_numeric(df[left_col], errors="coerce")
    right = pd.to_numeric(df[right_col], errors="coerce")

    if operator == "+":
        result = left + right
    elif operator == "-":
        result = left - right
    elif operator == "*":
        result = left * right
    elif operator == "/":
        result = np.where(right != 0, left / right, np.nan)
    else:
        raise ValueError("지원하지 않는 연산입니다.")

    df[new_col] = result
    return df


def create_transform_feature(df, source_col, new_col, transform_type):
    if source_col not in df.columns:
        raise ValueError(f"컬럼 {source_col}이 없습니다.")

    s = pd.to_numeric(df[source_col], errors="coerce")

    if transform_type == "log1p":
        values = np.log1p(s)
    elif transform_type == "sqrt":
        values = np.sqrt(np.maximum(s, 0))
    elif transform_type == "zscore":
        std = s.std(ddof=0)
        if std == 0 or np.isnan(std):
            values = pd.Series(0, index=s.index)
        else:
            values = (s - s.mean()) / std
    elif transform_type == "standardize":
        std = s.std(ddof=0)
        if std == 0 or np.isnan(std):
            values = pd.Series(0, index=s.index)
        else:
            values = (s - s.mean()) / std
    else:
        raise ValueError("지원하지 않는 변환입니다.")

    df[new_col] = values
    return df


# ============================================================
# 대용량 EDA 안정화 함수
# ============================================================
def make_plot_sample(df, max_rows=2000, random_state=42):
    """Plotly 브라우저 렌더링 병목을 줄이기 위한 시각화 전용 샘플."""
    if len(df) <= max_rows:
        return df.copy(), False
    return df.sample(n=max_rows, random_state=random_state).copy(), True


@st.cache_data(show_spinner=False)
def cached_corr(df_num, corr_cols):
    """상관계수 재계산 방지."""
    return df_num[list(corr_cols)].corr()


@st.cache_data(show_spinner=False)
def cached_anova(df, target, cat_cols, max_groups=30):
    """고카디널리티 범주형 변수를 제한하여 ANOVA 먹통 방지."""
    anova_rows = []
    skipped_rows = []

    for c in cat_cols:
        nunique = df[c].nunique(dropna=True)
        if nunique < 2:
            skipped_rows.append({"범주형 변수": c, "사유": "유효 범주 2개 미만"})
            continue
        if nunique > max_groups:
            skipped_rows.append({"범주형 변수": c, "사유": f"범주 수 {nunique}개 > 제한 {max_groups}개"})
            continue

        work = df[[c, target]].dropna().copy()
        groups = [g[target].values for _, g in work.groupby(c, observed=True) if len(g) >= 2]
        if len(groups) >= 2:
            f, p = stats.f_oneway(*groups)
            anova_rows.append({
                "범주형 변수": c,
                "범주 수": nunique,
                "F통계량": f,
                "p-value": p
            })
        else:
            skipped_rows.append({"범주형 변수": c, "사유": "표본 2개 이상인 그룹 부족"})

    anova_df = pd.DataFrame(anova_rows)
    if not anova_df.empty:
        anova_df = anova_df.sort_values("F통계량", ascending=False)

    skipped_df = pd.DataFrame(skipped_rows)
    return anova_df, skipped_df


def top_categories_for_plot(df, cat_col, y_col, top_n=20):
    """박스플롯에서 범주 수가 너무 많을 때 상위 범주만 사용."""
    cols = [cat_col, y_col]
    work = df[cols].dropna().copy()
    top_values = work[cat_col].astype(str).value_counts().head(top_n).index
    work[cat_col] = work[cat_col].astype(str)
    return work[work[cat_col].isin(top_values)].copy()


def show_target_distribution(df, target):
    if pd.api.types.is_numeric_dtype(df[target]):
        st.write("기술통계량")
        st.dataframe(df[target].describe().to_frame().T)

        fig = px.histogram(df, x=target, marginal="box", title=f"{target} 분포")
        st.plotly_chart(fig, width="stretch")

    else:
        st.write("기술통계량")
        st.dataframe(df[target].describe().to_frame().T)

        count_df = df[target].value_counts(dropna=False).reset_index()
        count_df.columns = [target, "count"]

        fig1 = px.bar(count_df, x=target, y="count", text="count", title=f"{target} 막대그래프")
        st.plotly_chart(fig1, width="stretch")

        fig2 = px.pie(count_df, names=target, values="count", title=f"{target} 파이차트")
        st.plotly_chart(fig2, width="stretch")


def sequential_merge(dfs, how):
    merged = dfs[0].copy()
    logs = []

    for i in range(1, len(dfs)):
        right = dfs[i].copy()
        common_cols = list(set(merged.columns).intersection(set(right.columns)))

        if len(common_cols) == 0:
            raise ValueError(f"{i+1}번째 데이터와 결합 가능한 공통 열이 없습니다.")

        before_left = merged.shape
        before_right = right.shape

        merged = pd.merge(merged, right, on=common_cols, how=how)

        logs.append({
            "단계": f"결합 {i}",
            "왼쪽 데이터 크기": before_left,
            "오른쪽 데이터 크기": before_right,
            "결합 키": ", ".join(common_cols),
            "결합 후 데이터 크기": merged.shape
        })

    return merged, pd.DataFrame(logs)


def limit_top_categories(train_df, test_df, cat_cols, top_n=5):
    train_df = train_df.copy()
    test_df = test_df.copy()
    top_info = {}

    for col in cat_cols:
        train_df[col] = train_df[col].astype(str)
        test_df[col] = test_df[col].astype(str)

        top_values = train_df[col].value_counts().head(top_n).index.tolist()
        top_info[col] = top_values

        train_df[col] = train_df[col].where(train_df[col].isin(top_values), "기타")
        test_df[col] = test_df[col].where(test_df[col].isin(top_values), "기타")

    return train_df, test_df, top_info


def preprocess_train_test(X_train, X_test):
    X_train = X_train.copy()
    X_test = X_test.copy()

    num_cols = get_numeric_cols(X_train)
    cat_cols = get_categorical_cols(X_train)

    X_train_limited, X_test_limited, top_info = limit_top_categories(X_train, X_test, cat_cols)

    scaler = StandardScaler()
    if num_cols:
        X_train_num = pd.DataFrame(
            scaler.fit_transform(X_train_limited[num_cols]),
            columns=num_cols,
            index=X_train_limited.index
        )
        X_test_num = pd.DataFrame(
            scaler.transform(X_test_limited[num_cols]),
            columns=num_cols,
            index=X_test_limited.index
        )
    else:
        X_train_num = pd.DataFrame(index=X_train_limited.index)
        X_test_num = pd.DataFrame(index=X_test_limited.index)

    if cat_cols:
        X_train_cat = pd.get_dummies(X_train_limited[cat_cols], dtype=int)
        X_test_cat = pd.get_dummies(X_test_limited[cat_cols], dtype=int)
        X_test_cat = X_test_cat.reindex(columns=X_train_cat.columns, fill_value=0)
    else:
        X_train_cat = pd.DataFrame(index=X_train_limited.index)
        X_test_cat = pd.DataFrame(index=X_test_limited.index)

    X_train_p = pd.concat([X_train_num, X_train_cat], axis=1)
    X_test_p = pd.concat([X_test_num, X_test_cat], axis=1)

    preprocess_info = {
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "scaler": scaler,
        "top_info": top_info,
        "columns": X_train_p.columns.tolist()
    }

    return X_train_p, X_test_p, preprocess_info


def preprocess_new_data(new_df, preprocess_info):
    new_df = new_df.copy()

    num_cols = preprocess_info["num_cols"]
    cat_cols = preprocess_info["cat_cols"]
    scaler = preprocess_info["scaler"]
    top_info = preprocess_info["top_info"]
    columns = preprocess_info["columns"]

    for col in num_cols + cat_cols:
        if col not in new_df.columns:
            new_df[col] = np.nan

    for col in num_cols:
        new_df[col] = pd.to_numeric(new_df[col], errors="coerce")
        new_df[col] = new_df[col].fillna(new_df[col].mean() if new_df[col].notna().any() else 0)

    for col in cat_cols:
        new_df[col] = new_df[col].astype(str)
        top_values = top_info.get(col, [])
        new_df[col] = new_df[col].where(new_df[col].isin(top_values), "기타")

    if num_cols:
        new_num = pd.DataFrame(
            scaler.transform(new_df[num_cols]),
            columns=num_cols,
            index=new_df.index
        )
    else:
        new_num = pd.DataFrame(index=new_df.index)

    if cat_cols:
        new_cat = pd.get_dummies(new_df[cat_cols], dtype=int)
    else:
        new_cat = pd.DataFrame(index=new_df.index)

    new_p = pd.concat([new_num, new_cat], axis=1)
    new_p = new_p.reindex(columns=columns, fill_value=0)

    return new_p


def regression_metrics(y_true, y_pred):
    return {
        "R2_Score": r2_score(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred) * 100
    }


def classification_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Recall_0": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "Recall_1": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "Precision_0": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "Precision_1": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "F1_score_0": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        "F1_score_1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "AUC": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan
    }



# ============================================================
# Counterfactual XAI 및 DoWhy 함수
# ============================================================
# 거리점수: (현재값- 후보값) / (현실 범위:최대-최소) 합산. 연속형은 정규화, 범주형은 1점 부여
def get_model_prediction(model, X_p, target_type):
    """전처리된 입력 X_p에 대해 예측값과 확률을 반환."""
    pred = model.predict(X_p)
    proba_1 = None
    if target_type == "범주형" and hasattr(model, "predict_proba"):
        try:
            prob = model.predict_proba(X_p)
            if prob.ndim == 2 and prob.shape[1] >= 2:
                proba_1 = prob[:, 1]
        except Exception:
            proba_1 = None
    return pred, proba_1


def generate_counterfactual_xai(
    input_df,
    model,
    preprocess_info,
    X_original,
    target_type,
    desired_rule,
    mutable_features,
    n_samples=2000,
    random_state=42,
    top_k=5
):
    """
    별도 패키지 없이 동작하는 단순 Counterfactual XAI 탐색 함수.
    - 입력값 주변의 후보 조건을 무작위 생성
    - 모델 예측이 목표 조건을 만족하는 후보만 선별
    - 현재 입력과 가까운 순서로 추천
    """
    rng = np.random.default_rng(random_state)
    base = input_df.iloc[0].copy()
    candidates = []

    num_cols = [c for c in X_original.columns if pd.api.types.is_numeric_dtype(X_original[c])]
    cat_cols = [c for c in X_original.columns if c not in num_cols]

    # 변수별 현실 범위/선택지를 학습 데이터 기준으로 정의
    num_ranges = {}
    for col in num_cols:
        s = pd.to_numeric(X_original[col], errors="coerce").dropna()
        if s.empty:
            continue
        q01, q99 = s.quantile(0.01), s.quantile(0.99)
        if q01 == q99:
            q01, q99 = s.min(), s.max()
        num_ranges[col] = (float(q01), float(q99), float(s.std()) if s.std() > 0 else 1.0)

    cat_values = {}
    for col in cat_cols:
        vals = X_original[col].astype(str).dropna().value_counts().head(20).index.tolist()
        if vals:
            cat_values[col] = vals

    mutable_features = [c for c in mutable_features if c in X_original.columns]
    if not mutable_features:
        return pd.DataFrame(), pd.DataFrame()

    for _ in range(int(n_samples)):
        row = base.copy()
        # 매 후보마다 변경할 변수 개수를 무작위 선택. 가까운 대안을 찾기 위해 일부 변수만 변경.
        max_change = max(1, min(len(mutable_features), 4))
        change_count = int(rng.integers(1, max_change + 1))
        changed = rng.choice(mutable_features, size=change_count, replace=False)

        for col in changed:
            if col in num_ranges:
                lo, hi, sd = num_ranges[col]
                # 70%는 현재값 주변, 30%는 전체 현실 범위에서 샘플링
                if rng.random() < 0.7:
                    val = float(row[col]) + rng.normal(0, sd * 0.5)
                    row[col] = float(np.clip(val, lo, hi))
                else:
                    row[col] = float(rng.uniform(lo, hi))
            elif col in cat_values:
                row[col] = rng.choice(cat_values[col])

        cand_df = pd.DataFrame([row])
        try:
            cand_p = preprocess_new_data(cand_df, preprocess_info)
            pred, proba_1 = get_model_prediction(model, cand_p, target_type)
        except Exception:
            continue

        if target_type == "연속형":
            yhat = float(pred[0])
            target_value = float(desired_rule["target_value"])
            direction = desired_rule["direction"]
            success = yhat <= target_value if direction == "이하" else yhat >= target_value
            score_value = yhat
            prob_value = np.nan
        else:
            target_class = desired_rule["target_class"]
            min_prob = float(desired_rule.get("min_prob", 0.5))
            yhat_label = pred[0]
            prob_value = float(proba_1[0]) if proba_1 is not None else np.nan
            # 이진분류에서 class 1을 목표로 할 때 확률 기준까지 함께 적용
            if str(target_class) in ["1", "1.0"] and not np.isnan(prob_value):
                success = (str(yhat_label) == str(target_class)) and (prob_value >= min_prob)
            else:
                success = str(yhat_label) == str(target_class)
            score_value = yhat_label

        if not success:
            continue

        # 현재 입력과의 거리 계산
        dist = 0.0
        changes = []
        for col in X_original.columns:
            before = base[col]
            after = row[col]
            if col in num_ranges:
                lo, hi, _ = num_ranges[col]
                denom = max(hi - lo, 1e-9)
                diff = abs(float(after) - float(before))
                if diff > 1e-9:
                    dist += diff / denom
                    changes.append(f"{col}: {float(before):.4g} → {float(after):.4g}")
            else:
                if str(before) != str(after):
                    dist += 1.0
                    changes.append(f"{col}: {before} → {after}")

        candidates.append({
            "거리점수": dist,
            "예측값": score_value,
            "예측확률_1": prob_value,
            "변경개수": len(changes),
            "변경 제안": " / ".join(changes) if changes else "변경 없음",
            **{c: row[c] for c in X_original.columns}
        })

    if not candidates:
        return pd.DataFrame(), pd.DataFrame()

    cf_df = pd.DataFrame(candidates).sort_values(["거리점수", "변경개수"]).head(int(top_k)).reset_index(drop=True)
    summary_cols = ["거리점수", "예측값", "예측확률_1", "변경개수", "변경 제안"]
    summary_df = cf_df[summary_cols].copy()
    return summary_df, cf_df


def patch_networkx_for_dowhy():
    """DoWhy와 NetworkX 버전 차이로 인한 d_separated 오류를 완화."""
    try:
        import networkx as nx
        if not hasattr(nx.algorithms, "d_separated"):
            try:
                from networkx.algorithms.d_separation import is_d_separator

                def _dowhy_compatible_d_separated(graph, x_nodes, y_nodes, z_nodes):
                    return is_d_separator(graph, set(x_nodes), set(y_nodes), set(z_nodes))

                nx.algorithms.d_separated = _dowhy_compatible_d_separated
            except Exception:
                pass
    except Exception:
        pass


def run_dowhy_analysis_numeric(causal_df, treatment_col, outcome_col, common_causes, method_name="backdoor.linear_regression"):
    """전처리된 숫자형 데이터 기준 DoWhy 4단계 실행."""
    patch_networkx_for_dowhy()
    from dowhy import CausalModel

    common_causes = [c for c in common_causes if c in causal_df.columns and c not in [treatment_col, outcome_col]]
    use_cols = list(dict.fromkeys([treatment_col, outcome_col] + common_causes))
    work = causal_df[use_cols].copy().dropna()

    # DoWhy에는 숫자형만 전달되도록 보장
    for c in use_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna()

    if len(work) < 30:
        raise ValueError("DoWhy 분석에 사용할 유효 데이터가 30행 미만입니다.")

    model = CausalModel(
        data=work,
        treatment=treatment_col,
        outcome=outcome_col,
        common_causes=common_causes
    )

    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(identified_estimand, method_name=method_name)

    refutations = {}
    for ref_name, kwargs in {
        "random_common_cause": {"method_name": "random_common_cause"},
        "placebo_treatment": {"method_name": "placebo_treatment_refuter", "placebo_type": "permute"},
        "data_subset": {"method_name": "data_subset_refuter", "subset_fraction": 0.8},
    }.items():
        try:
            refutations[ref_name] = model.refute_estimate(identified_estimand, estimate, **kwargs)
        except Exception as e:
            refutations[ref_name] = f"검증 실패: {e}"

    return {
        "data": work,
        "model": model,
        "identified_estimand": identified_estimand,
        "estimate": estimate,
        "effect_value": estimate.value,
        "refutations": refutations,
        "common_causes": common_causes,
    }


def fallback_causal_ols(causal_df, treatment_col, outcome_col, common_causes):
    """DoWhy 실행이 실패했을 때 앱 중단을 막기 위한 통제변수 OLS 대체 추정."""
    import statsmodels.api as sm
    common_causes = [c for c in common_causes if c in causal_df.columns and c not in [treatment_col, outcome_col]]
    cols = [treatment_col, outcome_col] + common_causes
    work = causal_df[cols].copy()
    for c in cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna()
    X = sm.add_constant(work[[treatment_col] + common_causes], has_constant="add")
    y = work[outcome_col]
    fit = sm.OLS(y, X).fit()
    return fit, work


# ============================================================
# Session State 초기화
# ============================================================
defaults = {
    "datasets": {},
    "merged_df": None,
    "working_df": None,
    "target": None,
    "target_type": None,
    "X_train": None,
    "X_test": None,
    "y_train": None,
    "y_test": None,
    "X_train_p": None,
    "X_test_p": None,
    "preprocess_info": None,
    "models": {},
    "train_result": None,
    "test_result": None,
    "best_model_name": None,
    "best_model": None,
    "predictions": {},
    "cf_xai_result": None,
    "dowhy_result": None
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ============================================================
# 화면 구성
# ============================================================
st.title("데이터 분석 절차 자동화 웹")

step = st.sidebar.radio(
    "분석 단계",
    [
        "1. 데이터 불러오기",
        "2. 데이터 결합하기",
        "3. 파생변수 생성하기",
        "4. 종속변수 분포확인",
        "5. 결측치 처리",
        "6. 이상치 처리",
        "7. 탐색적 분석",
        "8. 데이터 분할",
        "9. 모델링 전처리",
        "10. 모델 학습 및 평가",
        "11~12. 모델 비교 및 시각화",
        "13. 직접 입력 예측",
        "14. DoWhy 인과분석",
        "15. Counterfactual XAI",
        "16. 신규 데이터 예측"
    ]
)


# ============================================================
# 1. 데이터 불러오기
# ============================================================
if step == "1. 데이터 불러오기":
    st.header("1. 파일 업로드 및 데이터 확인")

    uploaded_files = st.file_uploader(
        "CSV 또는 XLSX 파일을 여러 개 업로드하세요.",
        type=["csv", "xlsx"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for file in uploaded_files:
            if file.name not in st.session_state.datasets:
                df, enc = read_uploaded_file(file)
                st.session_state.datasets[file.name] = {
                    "df": df,
                    "encoding": enc
                }

        for name, item in st.session_state.datasets.items():
            df = item["df"]
            st.subheader(f"{name} / 인코딩: {item['encoding']}")
            st.write("데이터 크기:", df.shape)
            st.dataframe(df.head())
            st.write("변수 정보")
            info_df = pd.DataFrame({
                "변수명": df.columns,
                "변수타입": df.dtypes.astype(str).values,
                "결측수": df.isna().sum().values
            })
            st.dataframe(info_df)
            st.write("기술통계량")
            st.dataframe(df.describe(include="all").T)
            download_csv_button(df, f"{name} 저장", name)


# ============================================================
# 2. 데이터 결합하기
# ============================================================
elif step == "2. 데이터 결합하기":
    st.header("2. 데이터 결합하기")

    if len(st.session_state.datasets) == 0:
        st.warning("먼저 데이터를 업로드하세요.")
        st.stop()

    names = list(st.session_state.datasets.keys())
    selected = st.multiselect("결합할 데이터 선택", names, default=names)

    axis_type = st.radio("결합 방향 선택", ["행 방향 결합", "열 방향 결합"])
    join_method = st.selectbox("열 방향 결합 방법", ["inner", "left", "right", "outer"])

    if st.button("데이터 결합 실행"):
        dfs = [st.session_state.datasets[n]["df"] for n in selected]

        if axis_type == "행 방향 결합":
            before = [{"데이터": n, "크기": dfs[i].shape} for i, n in enumerate(selected)]
            merged = pd.concat(dfs, axis=0, ignore_index=True)
            log_df = pd.DataFrame(before)
            log_df.loc[len(log_df)] = ["결합 후", merged.shape]

        else:
            merged, log_df = sequential_merge(dfs, how=join_method)

        st.session_state.merged_df = merged
        st.session_state.working_df = merged.copy()

        st.success("데이터 결합 완료")
        st.write("결합 로그")
        st.dataframe(log_df)
        st.write("결합 데이터")
        st.dataframe(merged.head())
        download_csv_button(merged, "결합 데이터 CSV 저장", "merged_data.csv")

    if st.session_state.merged_df is not None:
        st.subheader("현재 결합 데이터")
        st.write(st.session_state.merged_df.shape)
        st.dataframe(st.session_state.merged_df.head())


# ============================================================
# 3. 파생변수 생성하기
# ============================================================
elif step == "3. 파생변수 생성하기":
    st.header("3. 파생변수 생성하기")

    df = st.session_state.working_df
    if df is None:
        st.warning("먼저 데이터를 결합하거나 업로드하세요.")
        st.stop()

    num_cols = get_numeric_cols(df)
    cat_cols = get_categorical_cols(df)
    method = st.selectbox(
        "파생변수 생성 방식",
        ["binning", "dummy", "tukey", "사칙연산", "transform"],
        format_func=lambda x: {
            "binning": "1) binning",
            "dummy": "2) dummy",
            "tukey": "3) tukey",
            "사칙연산": "4) 사칙연산",
            "transform": "5) transform"
        }[x]
    )

    new_col = st.text_input("파생변수명", value="new_feature")

    if method == "binning":
        if not num_cols:
            st.warning("수치형 컬럼이 없어 binning을 수행할 수 없습니다.")
            st.stop()
        source_col = st.selectbox("기준 연속형 변수", num_cols)
        bin_method = st.radio("구간 분할 방식", ["등간격", "분위수"], horizontal=True)
        n_bins = st.number_input("구간 수", min_value=2, max_value=10, value=3, step=1)
        if st.button("파생변수 생성"):
            try:
                df = create_binning_feature(df, source_col, new_col, n_bins=n_bins, method=bin_method)
                st.session_state.working_df = df
                st.success(f"{new_col} 생성 완료")
                st.dataframe(df[[source_col, new_col]].head())
            except Exception as e:
                st.error(f"binning 생성 실패: {e}")

    elif method == "dummy":
        if not cat_cols:
            st.warning("범주형 컬럼이 없어 dummy 생성이 불가능합니다.")
            st.stop()
        source_col = st.selectbox("기준 범주형 변수", cat_cols)
        drop_first = st.checkbox("첫 번째 범주 제외", value=False)
        if st.button("파생변수 생성"):
            try:
                df = create_dummy_feature(df, source_col, new_col, drop_first=drop_first)
                st.session_state.working_df = df
                st.success(f"{new_col} 기반 dummy 생성 완료")
                st.dataframe(df[[source_col] + [c for c in df.columns if c.startswith(new_col)]].head())
            except Exception as e:
                st.error(f"dummy 생성 실패: {e}")

    elif method == "tukey":
        if not num_cols:
            st.warning("수치형 컬럼이 없어 Tukey 기반 생성이 불가능합니다.")
            st.stop()
        source_col = st.selectbox("기준 연속형 변수", num_cols)
        power = st.slider("Tukey's ladder of power", min_value=-3, max_value=3, value=1, step=1)
        if st.button("파생변수 생성"):
            try:
                df = create_tukey_feature(df, source_col, new_col, power=power)
                st.session_state.working_df = df
                st.success(f"{new_col} 생성 완료")
                st.dataframe(df[[source_col, new_col]].head())
            except Exception as e:
                st.error(f"Tukey 생성 실패: {e}")

    elif method == "사칙연산":
        if len(num_cols) < 2:
            st.warning("사칙연산을 위해 최소 2개의 수치형 컬럼이 필요합니다.")
            st.stop()
        left_col = st.selectbox("왼쪽 변수", num_cols)
        right_col = st.selectbox("오른쪽 변수", num_cols, index=1 if len(num_cols) > 1 else 0)
        operator = st.selectbox("연산", ["+", "-", "*", "/"])
        if st.button("파생변수 생성"):
            try:
                df = create_arithmetic_feature(df, left_col, right_col, operator, new_col)
                st.session_state.working_df = df
                st.success(f"{new_col} 생성 완료")
                st.dataframe(df[[left_col, right_col, new_col]].head())
            except Exception as e:
                st.error(f"사칙연산 생성 실패: {e}")

    elif method == "transform":
        if not num_cols:
            st.warning("수치형 컬럼이 없어 transform을 수행할 수 없습니다.")
            st.stop()
        source_col = st.selectbox("기준 연속형 변수", num_cols)
        transform_type = st.selectbox("변환 방식", ["log1p", "sqrt", "zscore", "standardize"])
        if st.button("파생변수 생성"):
            try:
                df = create_transform_feature(df, source_col, new_col, transform_type)
                st.session_state.working_df = df
                st.success(f"{new_col} 생성 완료")
                st.dataframe(df[[source_col, new_col]].head())
            except Exception as e:
                st.error(f"transform 생성 실패: {e}")

    download_csv_button(df, "현재 데이터 저장", "derived_data.csv")


# ============================================================
# 4. 종속변수 분포확인
# ============================================================
elif step == "4. 종속변수 분포확인":
    st.header("4. 종속변수 분포확인")

    df = st.session_state.working_df
    if df is None:
        st.warning("데이터가 없습니다.")
        st.stop()

    target = st.selectbox("종속변수 선택", df.columns)
    target_type = st.radio("종속변수 타입", ["연속형", "범주형"])

    st.session_state.target = target
    st.session_state.target_type = target_type

    show_target_distribution(df, target)
    download_csv_button(df, "현재 데이터 저장", "target_checked_data.csv")


# ============================================================
# 5. 결측치 처리
# ============================================================
elif step == "5. 결측치 처리":
    st.header("5. 결측치 처리")

    df = st.session_state.working_df
    if df is None:
        st.warning("데이터가 없습니다.")
        st.stop()

    missing_df = pd.DataFrame({
        "변수명": df.columns,
        "결측수": df.isna().sum().values,
        "결측비율(%)": (df.isna().mean() * 100).round(2).values
    })

    st.dataframe(missing_df)

    target_col = st.selectbox("결측 처리할 변수 선택", df.columns)
    method = st.radio("처리 방법", ["결측 행 삭제", "대표값 대체"])

    if method == "대표값 대체":
        if pd.api.types.is_numeric_dtype(df[target_col]):
            rep_method = st.selectbox("대표값", ["평균", "중앙값", "최빈값"])
        else:
            rep_method = st.selectbox("대표값", ["최빈값"])

    if st.button("결측치 처리 실행"):
        before_shape = df.shape

        if method == "결측 행 삭제":
            df = df[df[target_col].notna()].copy()
        else:
            if rep_method == "평균":
                value = df[target_col].mean()
            elif rep_method == "중앙값":
                value = df[target_col].median()
            else:
                value = df[target_col].mode(dropna=True)[0]

            df[target_col] = df[target_col].fillna(value)

        st.session_state.working_df = df

        st.success("결측치 처리 완료")
        st.write("처리 전 크기:", before_shape)
        st.write("처리 후 크기:", df.shape)
        st.dataframe(df.head())

    download_csv_button(df, "결측치 처리 데이터 저장", "missing_processed_data.csv")


# ============================================================
# 6. 이상치 처리
# ============================================================
elif step == "6. 이상치 처리":
    st.header("6. 이상치 처리")

    df = st.session_state.working_df
    if df is None:
        st.warning("데이터가 없습니다.")
        st.stop()

    num_cols = get_numeric_cols(df)
    col = st.selectbox("이상치 확인 변수", num_cols)

    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outlier_mask = (df[col] < lower) | (df[col] > upper)
    outlier_count = outlier_mask.sum()
    outlier_ratio = outlier_count / len(df) * 100

    st.write(f"Q1: {q1:.4f}, Q3: {q3:.4f}, IQR: {iqr:.4f}")
    st.write(f"하한: {lower:.4f}, 상한: {upper:.4f}")
    st.write(f"이상치 수: {outlier_count}, 이상치 비율: {outlier_ratio:.2f}%")

    fig1 = px.histogram(df, x=col, marginal="box", title=f"{col} 이상치 확인")
    st.plotly_chart(fig1, width="stretch")

    method = st.radio("이상치 처리 방법", ["처리하지 않음", "이상치 행 제거", "정상 범위 최소/최대값으로 대체", "직접 범위 지정"])

    if method == "직접 범위 지정":
        custom_lower = st.number_input("하한값", value=float(lower))
        custom_upper = st.number_input("상한값", value=float(upper))

    if st.button("이상치 처리 실행"):
        before = df.shape

        if method == "이상치 행 제거":
            df = df[~outlier_mask].copy()

        elif method == "정상 범위 최소/최대값으로 대체":
            normal = df.loc[~outlier_mask, col]
            normal_min = normal.min()
            normal_max = normal.max()
            df.loc[df[col] < lower, col] = normal_min
            df.loc[df[col] > upper, col] = normal_max

        elif method == "직접 범위 지정":
            df[col] = df[col].clip(custom_lower, custom_upper)

        st.session_state.working_df = df

        st.success("이상치 처리 완료")
        st.write("처리 전 크기:", before)
        st.write("처리 후 크기:", df.shape)

        fig2 = px.histogram(df, x=col, marginal="box", title=f"{col} 이상치 처리 후")
        st.plotly_chart(fig2, width="stretch")

    download_csv_button(df, "이상치 처리 데이터 저장", "outlier_processed_data.csv")


# ============================================================
# 7. 탐색적 분석
# ============================================================
elif step == "7. 탐색적 분석":
    st.header("7. 탐색적 분석")

    df = st.session_state.working_df
    if df is None:
        st.warning("데이터가 없습니다.")
        st.stop()

    num_cols = get_numeric_cols(df)
    cat_cols = get_categorical_cols(df)

    st.info(
        "대용량 데이터에서 브라우저가 멈추지 않도록 시각화는 샘플 데이터로 그리고, "
        "상관분석/ANOVA는 필요한 변수만 선택해 계산합니다. 원본 데이터는 변경하지 않습니다."
    )

    with st.expander("대용량 EDA 옵션", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            max_plot_rows = st.number_input(
                "그래프 최대 행 수",
                min_value=300,
                max_value=10000,
                value=min(2000, max(300, len(df))),
                step=100,
                help="Plotly 그래프에 전달할 최대 행 수입니다. 데이터가 많을수록 1,000~3,000 권장"
            )
        with c2:
            max_hover_cols = st.number_input(
                "hover 표시 변수 수",
                min_value=0,
                max_value=max(0, len(df.columns)),
                value=min(5, len(df.columns)),
                step=1,
                help="hover_data=df.columns 전체 전달이 주요 병목입니다. 꼭 필요한 변수만 표시하세요."
            )
        with c3:
            top_n_categories = st.number_input(
                "박스플롯 상위 범주 수",
                min_value=3,
                max_value=100,
                value=20,
                step=1
            )

    plot_df, sampled = make_plot_sample(df, int(max_plot_rows))
    if sampled:
        st.caption(f"그래프는 전체 {len(df):,}행 중 {len(plot_df):,}행을 무작위 샘플링하여 표시합니다.")

    st.subheader("산점도")
    if len(num_cols) < 2:
        st.warning("산점도를 그리려면 연속형 변수가 2개 이상 필요합니다.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            x_var = st.selectbox("X축 연속형 변수", num_cols, key="scatter_x")
        with col2:
            y_var = st.selectbox("Y축 연속형 변수", num_cols, index=1 if len(num_cols) > 1 else 0, key="scatter_y")

        available_hover = [c for c in df.columns if c not in [x_var, y_var]]
        default_hover = available_hover[:int(max_hover_cols)]
        hover_cols = st.multiselect(
            "산점도 hover 변수 선택",
            available_hover,
            default=default_hover,
            help="전체 변수를 hover에 넣으면 Plotly JSON이 급격히 커져 화면이 멈출 수 있습니다."
        )

        scatter_cols = [x_var, y_var] + hover_cols
        scatter_df = plot_df[scatter_cols].dropna(subset=[x_var, y_var])
        fig = px.scatter(
            scatter_df,
            x=x_var,
            y=y_var,
            hover_data=hover_cols,
            render_mode="webgl",
            title=f"{x_var} vs {y_var}"
        )
        st.plotly_chart(fig, width="stretch")

    st.subheader("박스플롯")
    if cat_cols and num_cols:
        bx = st.selectbox("X축 범주형 변수", cat_cols)
        by = st.selectbox("Y축 연속형 변수", num_cols)

        box_df = top_categories_for_plot(plot_df, bx, by, int(top_n_categories))
        if box_df.empty:
            st.warning("박스플롯을 그릴 수 있는 유효 데이터가 없습니다.")
        else:
            original_nunique = df[bx].nunique(dropna=True)
            if original_nunique > int(top_n_categories):
                st.caption(f"'{bx}'의 전체 범주 {original_nunique:,}개 중 빈도 상위 {int(top_n_categories)}개만 표시합니다.")

            fig = px.box(
                box_df,
                x=bx,
                y=by,
                points=False,
                title=f"{bx}별 {by}"
            )
            st.plotly_chart(fig, width="stretch")
    else:
        st.warning("박스플롯을 그리려면 범주형 변수와 연속형 변수가 각각 1개 이상 필요합니다.")

    st.subheader("상관분석 히트맵")
    if len(num_cols) < 2:
        st.warning("상관분석을 하려면 연속형 변수가 2개 이상 필요합니다.")
    else:
        corr_cols = st.multiselect(
            "상관분석 변수 선택",
            num_cols,
            default=num_cols[:min(8, len(num_cols))]
        )

        if len(corr_cols) >= 2:
            df_num = df[corr_cols].apply(pd.to_numeric, errors="coerce")
            corr = cached_corr(df_num, tuple(corr_cols))

            fig = px.imshow(
                corr,
                text_auto=".2f",
                aspect="auto",
                title="상관계수 히트맵"
            )
            fig.update_traces(textfont_size=10)
            st.plotly_chart(fig, width="stretch")

            target = st.session_state.target
            if target in corr_cols:
                corr_result = (
                    corr[target]
                    .drop(target)
                    .abs()
                    .sort_values(ascending=False)
                    .reset_index()
                )
                corr_result.columns = ["변수명", f"{target} 기준 절대상관계수"]
                st.markdown(f"##### `{target}` 기준 상관분석 결과")
                st.dataframe(corr_result, width="stretch")
        else:
            st.caption("상관분석 변수는 2개 이상 선택하세요.")

    st.subheader("분산분석 (ANOVA)")
    if st.session_state.target and st.session_state.target in df.columns and cat_cols:
        target = st.session_state.target
        if pd.api.types.is_numeric_dtype(df[target]):
            selected_cat_cols = st.multiselect(
                "ANOVA 범주형 변수 선택",
                cat_cols,
                default=cat_cols[:min(5, len(cat_cols))],
                key="anova_cat_cols"
            )
            max_anova_groups = st.number_input(
                "ANOVA 변수별 최대 범주 수",
                min_value=2,
                max_value=200,
                value=30,
                step=1,
                key="anova_max_groups",
                help="범주가 너무 많은 변수는 계산에서 제외됩니다."
            )

            if st.button("ANOVA 계산 실행", key="anova_run"):
                if selected_cat_cols:
                    anova_df, skipped_df = cached_anova(
                        df[selected_cat_cols + [target]].copy(),
                        target,
                        tuple(selected_cat_cols),
                        int(max_anova_groups)
                    )
                    if not anova_df.empty:
                        st.dataframe(anova_df, width="stretch")
                    else:
                        st.warning("계산 가능한 ANOVA 결과가 없습니다.")

                    if not skipped_df.empty:
                        with st.expander("ANOVA 제외 변수 보기"):
                            st.dataframe(skipped_df, width="stretch")
                else:
                    st.warning("ANOVA에 사용할 범주형 변수를 선택하세요.")
        else:
            st.caption("ANOVA는 종속변수가 연속형일 때 계산합니다.")
    else:
        st.caption("ANOVA를 계산하려면 먼저 종속변수를 선택하고 범주형 변수가 있어야 합니다.")


# ============================================================
# 8. 데이터 분할
# ============================================================
elif step == "8. 데이터 분할":
    st.header("8. 학습/평가 데이터 분할")

    df = st.session_state.working_df
    if df is None:
        st.warning("데이터가 없습니다.")
        st.stop()

    target = st.selectbox(
        "종속변수 선택",
        df.columns,
        index=list(df.columns).index(st.session_state.target) if st.session_state.target in df.columns else 0
    )

    features = st.multiselect(
        "독립변수 선택",
        [c for c in df.columns if c != target],
        default=[c for c in df.columns if c != target]
    )

    target_type = st.radio("종속변수 타입", ["연속형", "범주형"], index=0 if pd.api.types.is_numeric_dtype(df[target]) else 1)

    test_size = st.slider("평가 데이터 비율", 0.1, 0.5, 0.3, 0.05)

    st.subheader("분할 전 종속변수 분포")
    show_target_distribution(df, target)

    if st.button("데이터 분할 실행"):
        data = df[features + [target]].dropna().copy()
        X = data[features]
        y = data[target]

        stratify = y if target_type == "범주형" and y.nunique() > 1 else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=42,
            stratify=stratify
        )

        st.session_state.target = target
        st.session_state.target_type = target_type
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test

        st.success("데이터 분할 완료")

    if st.session_state.X_train is not None:
        st.subheader("학습 데이터")
        st.dataframe(st.session_state.X_train.head())
        st.write(st.session_state.X_train.shape)

        st.subheader("평가 데이터")
        st.dataframe(st.session_state.X_test.head())
        st.write(st.session_state.X_test.shape)

        st.subheader("분할 후 학습 종속변수 분포")
        show_target_distribution(pd.DataFrame({st.session_state.target: st.session_state.y_train}), st.session_state.target)

        st.subheader("분할 후 평가 종속변수 분포")
        show_target_distribution(pd.DataFrame({st.session_state.target: st.session_state.y_test}), st.session_state.target)


# ============================================================
# 9. 모델링 전처리
# ============================================================
elif step == "9. 모델링 전처리":
    st.header("9. 표준화 및 더미변수 처리")

    if st.session_state.X_train is None:
        st.warning("먼저 데이터 분할을 실행하세요.")
        st.stop()

    if st.button("전처리 실행"):
        X_train_p, X_test_p, preprocess_info = preprocess_train_test(
            st.session_state.X_train,
            st.session_state.X_test
        )

        st.session_state.X_train_p = X_train_p
        st.session_state.X_test_p = X_test_p
        st.session_state.preprocess_info = preprocess_info

        st.success("전처리 완료")

    if st.session_state.X_train_p is not None:
        st.subheader("전처리된 학습 데이터")
        st.dataframe(st.session_state.X_train_p.head())

        st.subheader("전처리된 평가 데이터")
        st.dataframe(st.session_state.X_test_p.head())

        download_csv_button(st.session_state.X_train_p, "전처리 학습 데이터 저장", "X_train_processed.csv")
        download_csv_button(st.session_state.X_test_p, "전처리 평가 데이터 저장", "X_test_processed.csv")


# ============================================================
# 10. 모델 학습 및 평가
# ============================================================
elif step == "10. 모델 학습 및 평가":
    st.header("10. 모델 학습 및 평가")

    if st.session_state.X_train_p is None:
        st.warning("먼저 모델링 전처리를 실행하세요.")
        st.stop()

    target_type = st.session_state.target_type
    X_train = st.session_state.X_train_p
    X_test = st.session_state.X_test_p
    y_train = st.session_state.y_train
    y_test = st.session_state.y_test

    if target_type == "연속형":
        models = {
            "회귀분석": LinearRegression(),
            "의사결정나무": DecisionTreeRegressor(random_state=42, max_depth=5),
            "랜덤포레스트": RandomForestRegressor(random_state=42, n_estimators=200),
            "인공신경망": MLPRegressor(random_state=42, max_iter=1000),
            "릿지(Ridge)": Ridge(alpha=1.0, random_state=42),
            "라쏘(Lasso)": Lasso(alpha=1.0, random_state=42)
        }
        if XGB_AVAILABLE:
            models["XGB"] = XGBRegressor(random_state=42, objective="reg:squarederror")

    else:
        models = {
            "로지스틱회귀분석": LogisticRegression(max_iter=1000, random_state=42),
            "의사결정나무": DecisionTreeClassifier(random_state=42, max_depth=5),
            "랜덤포레스트": RandomForestClassifier(random_state=42, n_estimators=200),
            "인공신경망": MLPClassifier(random_state=42, max_iter=1000)
        }
        if XGB_AVAILABLE:
            models["XGB"] = XGBClassifier(random_state=42, eval_metric="logloss")

    selected_models = st.multiselect("학습할 모델 선택", list(models.keys()), default=list(models.keys()))

    if st.button("모델 학습 실행"):
        trained = {}
        preds = {}
        train_rows = []
        test_rows = {}

        for name in selected_models:
            model = models[name]
            model.fit(X_train, y_train)
            trained[name] = model

            pred_train = model.predict(X_train)
            pred_test = model.predict(X_test)

            if target_type == "연속형":
                train_metric = {"모델명": name, **regression_metrics(y_train, pred_train)}
                test_metric = {"모델명": name, **regression_metrics(y_test, pred_test)}
                preds[name] = {"train_pred": pred_train, "test_pred": pred_test}

            else:
                if hasattr(model, "predict_proba"):
                    prob_train = model.predict_proba(X_train)[:, 1]
                    prob_test = model.predict_proba(X_test)[:, 1]
                else:
                    prob_train = pred_train
                    prob_test = pred_test

                train_metric = {"모델명": name, **classification_metrics(y_train, pred_train, prob_train)}
                test_metric = {"모델명": name, **classification_metrics(y_test, pred_test, prob_test)}
                preds[name] = {
                    "train_pred": pred_train,
                    "test_pred": pred_test,
                    "train_prob": prob_train,
                    "test_prob": prob_test
                }

            train_rows.append(train_metric)
            test_rows[name] = test_metric

        test_result = pd.DataFrame(list(test_rows.values()))
        train_result = pd.DataFrame(train_rows)

        if target_type == "연속형":
            best_name = test_result.sort_values(["RMSE", "MAPE"], ascending=True).iloc[0]["모델명"]
        else:
            best_name = test_result.sort_values(["AUC", "Accuracy"], ascending=False).iloc[0]["모델명"]

        st.session_state.models = trained
        st.session_state.predictions = preds
        st.session_state.train_result = train_result
        st.session_state.test_result = test_result
        st.session_state.best_model_name = best_name
        st.session_state.best_model = trained[best_name]

        st.success(f"모델 학습 완료 / 최적 모델: {best_name}")

    if st.session_state.train_result is not None:
        st.subheader("학습용 성능평가")
        st.dataframe(st.session_state.train_result)

        st.subheader("평가용 성능평가")
        st.dataframe(st.session_state.test_result)

        download_csv_button(st.session_state.train_result, "학습 성능 저장", "train_metrics.csv")
        download_csv_button(st.session_state.test_result, "평가 성능 저장", "test_metrics.csv")


# ============================================================
# 11~12. 모델 비교 및 시각화
# ============================================================
elif step == "11~12. 모델 비교 및 시각화":
    st.header("11~12. 모델링 결과 비교 및 시각화")

    if st.session_state.test_result is None:
        st.warning("먼저 모델 학습을 실행하세요.")
        st.stop()

    st.write("최적 모델:", st.session_state.best_model_name)
    st.dataframe(st.session_state.test_result)

    target_type = st.session_state.target_type
    best_name = st.session_state.best_model_name

    if target_type == "연속형":
        y_test = st.session_state.y_test
        pred = st.session_state.predictions[best_name]["test_pred"]

        plot_df = st.session_state.X_test.copy()
        plot_df["실제값"] = y_test.values
        plot_df["예측값"] = pred

        fig = px.scatter(
            plot_df,
            x="실제값",
            y="예측값",
            hover_data=plot_df.columns,
            title=f"{best_name} 실제값 vs 예측값"
        )

        fig.add_trace(
            go.Scatter(
                x=[plot_df["실제값"].min(), plot_df["실제값"].max()],
                y=[plot_df["실제값"].min(), plot_df["실제값"].max()],
                mode="lines",
                name="정확 예측선"
            )
        )

        st.plotly_chart(fig, width="stretch")

    else:
        y_test = st.session_state.y_test

        fig = go.Figure()

        for name, pred_info in st.session_state.predictions.items():
            prob = pred_info["test_prob"]
            fpr, tpr, _ = roc_curve(y_test, prob)
            auc = roc_auc_score(y_test, prob)

            fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"{name} AUC={auc:.4f}"
                )
            )

        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random",
                line=dict(dash="dash")
            )
        )

        fig.update_layout(
            title="ROC Curve 비교",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate"
        )

        st.plotly_chart(fig, width="stretch")


# ============================================================
# 13. 직접 입력 예측
# ============================================================
elif step == "13. 직접 입력 예측":
    st.header("13. 최적 모델 직접 입력 예측")

    if st.session_state.best_model is None:
        st.warning("먼저 모델 학습을 실행하세요.")
        st.stop()

    X_original = st.session_state.X_train
    feature_cols = X_original.columns.tolist()

    left, right = st.columns(2)

    input_data = {}

    with left:
        st.subheader("독립변수 입력")

        for col in feature_cols:
            if pd.api.types.is_numeric_dtype(X_original[col]):
                min_v = float(X_original[col].min())
                max_v = float(X_original[col].max())

                slider_min = min_v * 2 if min_v < 0 else min_v / 2
                slider_max = max_v * 2 if max_v > 0 else max_v / 2
                default_v = float(X_original[col].median())

                val = st.number_input(
                    f"{col}",
                    min_value=slider_min,
                    max_value=slider_max,
                    value=default_v,
                    key=f"input_{col}"
                )
                input_data[col] = val

            else:
                options = X_original[col].astype(str).unique().tolist()
                val = st.selectbox(col, options, key=f"input_{col}")
                input_data[col] = val

        input_df = pd.DataFrame([input_data])

        download_excel_button(input_df, "입력값 엑셀 다운로드", "prediction_input_template.xlsx")

    with right:
        st.subheader("예측 결과")

        input_p = preprocess_new_data(input_df, st.session_state.preprocess_info)
        model = st.session_state.best_model

        pred = model.predict(input_p)

        result_df = input_df.copy()
        result_df["예측값"] = pred

        if st.session_state.target_type == "범주형" and hasattr(model, "predict_proba"):
            result_df["예측확률_1"] = model.predict_proba(input_p)[:, 1]

        st.dataframe(result_df)
        download_csv_button(result_df, "예측 결과 저장", "single_prediction_result.csv")



# ============================================================
# 14. DoWhy 인과분석
# ============================================================
elif step == "14. DoWhy 인과분석":
    st.header("14. DoWhy 인과분석: 모델 독립변수별 인과효과 추정")

    if st.session_state.X_train_p is None or st.session_state.y_train is None:
        st.warning("먼저 8~9단계에서 데이터 분할과 모델링 전처리를 실행하세요.")
        st.stop()

    st.info(
        "모델에 사용된 전처리 독립변수 중 하나를 처치변수로 선택하여 DoWhy의 Model → Identify → Estimate → Refute 절차를 실행합니다. "
        "전처리 데이터 기준이므로 연속형 변수는 표준화 1단위 효과, 더미 변수는 기준범주 대비 0→1 변화 효과로 해석하세요."
    )

    X_all_p = pd.concat([st.session_state.X_train_p, st.session_state.X_test_p], axis=0)
    y_all = pd.concat([pd.Series(st.session_state.y_train), pd.Series(st.session_state.y_test)], axis=0).reset_index(drop=True)
    X_all_p = X_all_p.reset_index(drop=True)

    outcome_col = "__outcome__"
    causal_df = X_all_p.copy()

    # outcome이 문자형이면 코드화하여 DoWhy에 전달
    if st.session_state.target_type == "범주형":
        y_numeric, class_names = pd.factorize(y_all.astype(str))
        causal_df[outcome_col] = y_numeric
        class_map = pd.DataFrame({"원래 클래스": class_names, "DoWhy 숫자코드": range(len(class_names))})
        with st.expander("범주형 종속변수 숫자코드 매핑"):
            st.dataframe(class_map, width="stretch")
    else:
        causal_df[outcome_col] = pd.to_numeric(y_all, errors="coerce")

    feature_options = X_all_p.columns.tolist()

    col1, col2 = st.columns([1, 1])
    with col1:
        treatment_col = st.selectbox("처치변수 선택", feature_options)
        default_causes = [c for c in feature_options if c != treatment_col]
        max_default = min(10, len(default_causes))
        selected_common_causes = st.multiselect(
            "통제변수 선택",
            default_causes,
            default=default_causes[:max_default],
            help="처치변수와 결과변수 모두에 영향을 줄 수 있는 변수들을 선택하세요. 너무 많이 선택하면 계산이 느려질 수 있습니다."
        )

    with col2:
        method_name = st.selectbox(
            "추정 방법",
            [
                "backdoor.linear_regression",
                "backdoor.propensity_score_matching",
                "backdoor.propensity_score_stratification",
                "backdoor.propensity_score_weighting"
            ],
            index=0,
            help="연속형 처치변수는 linear_regression을 권장합니다. 더미/이진 처치변수는 propensity 계열도 사용할 수 있습니다."
        )
        st.write("분석 데이터 크기", causal_df[[treatment_col, outcome_col] + selected_common_causes].dropna().shape)
        st.caption(f"Outcome: {st.session_state.target} → DoWhy 내부 컬럼 {outcome_col}")

    with st.expander("DoWhy 분석 구조 확인", expanded=True):
        st.markdown(f"""
        - **Model**: `{treatment_col}` → `{st.session_state.target}` 인과효과 가정
        - **Identify**: 선택한 통제변수로 backdoor 경로 차단
        - **Estimate**: `{method_name}`으로 평균처치효과 추정
        - **Refute**: random common cause, placebo treatment, data subset으로 견고성 검증
        """)
        if selected_common_causes:
            st.write("통제변수", selected_common_causes)

    if st.button("선택 변수로 DoWhy 실행", type="primary"):
        try:
            with st.spinner("DoWhy 4단계 분석 실행 중입니다..."):
                result = run_dowhy_analysis_numeric(
                    causal_df=causal_df,
                    treatment_col=treatment_col,
                    outcome_col=outcome_col,
                    common_causes=selected_common_causes,
                    method_name=method_name
                )
            st.session_state.dowhy_result = result
            st.success("DoWhy 분석 완료")

            c1, c2 = st.columns(2)
            with c1:
                st.metric("추정 인과효과", f"{float(result['effect_value']):.6f}")
            with c2:
                st.metric("사용 데이터 행 수", f"{len(result['data']):,}")

            st.subheader("1단계 Model")
            st.write(f"처치변수 `{treatment_col}`가 결과변수 `{st.session_state.target}`에 미치는 효과를 가정했습니다.")

            st.subheader("2단계 Identify")
            st.text(str(result["identified_estimand"]))

            st.subheader("3단계 Estimate")
            st.text(str(result["estimate"]))

            st.subheader("4단계 Refute")
            for name, refute_result in result["refutations"].items():
                with st.expander(name, expanded=True):
                    st.text(str(refute_result))

            report_df = pd.DataFrame([{
                "처치변수": treatment_col,
                "결과변수": st.session_state.target,
                "추정방법": method_name,
                "인과효과": result["effect_value"],
                "통제변수": ", ".join(selected_common_causes),
                "사용행수": len(result["data"])
            }])
            download_csv_button(report_df, "DoWhy 결과 요약 저장", "dowhy_result_summary.csv")

        except ImportError:
            st.error("DoWhy가 설치되어 있지 않습니다. 터미널에서 `python -m pip install dowhy`를 실행하세요.")
        except Exception as e:
            st.error(f"DoWhy 실행 중 오류가 발생했습니다: {e}")
            st.warning("대체 방법으로 통제변수 OLS 추정을 실행합니다. 이는 DoWhy의 Refute 검증을 포함하지 않는 보조 결과입니다.")
            try:
                fit, work = fallback_causal_ols(causal_df, treatment_col, outcome_col, selected_common_causes)
                st.metric("OLS 대체 추정 효과", f"{float(fit.params[treatment_col]):.6f}")
                st.text(fit.summary().as_text())
            except Exception as e2:
                st.error(f"OLS 대체 추정도 실패했습니다: {e2}")


# ============================================================
# 15. Counterfactual XAI
# ============================================================
elif step == "15. Counterfactual XAI":
    st.header("15. Counterfactual XAI: 목표 예측을 만족하는 최소 변경 조건 탐색")

    if st.session_state.best_model is None or st.session_state.preprocess_info is None:
        st.warning("먼저 8~10단계에서 데이터 분할, 전처리, 모델 학습을 실행하세요.")
        st.stop()

    X_original = st.session_state.X_train
    model = st.session_state.best_model
    target_type = st.session_state.target_type
    feature_cols = X_original.columns.tolist()

    st.info(
        "Counterfactual XAI는 현재 입력값에서 어떤 독립변수를 얼마나 바꾸면 원하는 예측 결과에 도달하는지 제안합니다. "
        "여기서는 별도 패키지 없이 학습 데이터의 현실 범위 안에서 후보 조건을 생성하고, 목표를 만족하는 가까운 대안을 추천합니다."
    )

    left, right = st.columns([1, 1.2])

    input_data = {}
    with left:
        st.subheader("현재 조건 입력")
        for col in feature_cols:
            if pd.api.types.is_numeric_dtype(X_original[col]):
                min_v = float(X_original[col].min())
                max_v = float(X_original[col].max())
                default_v = float(X_original[col].median())
                val = st.number_input(
                    f"{col}",
                    min_value=min_v,
                    max_value=max_v,
                    value=default_v,
                    key=f"cf_input_{col}"
                )
                input_data[col] = val
            else:
                options = X_original[col].astype(str).dropna().unique().tolist()
                if not options:
                    options = [""]
                val = st.selectbox(col, options, key=f"cf_input_{col}")
                input_data[col] = val

        input_df = pd.DataFrame([input_data])
        input_p = preprocess_new_data(input_df, st.session_state.preprocess_info)
        base_pred, base_proba_1 = get_model_prediction(model, input_p, target_type)

        st.subheader("현재 예측")
        if target_type == "연속형":
            st.metric("현재 예측값", f"{float(base_pred[0]):.4f}")
        else:
            st.metric("현재 예측 클래스", str(base_pred[0]))
            if base_proba_1 is not None:
                st.metric("현재 예측확률_1", f"{float(base_proba_1[0]):.4f}")

    with right:
        st.subheader("Counterfactual 목표 설정")

        mutable_features = st.multiselect(
            "변경 가능한 독립변수 선택",
            feature_cols,
            default=feature_cols[:min(8, len(feature_cols))],
            help="현장에서 실제로 조정 가능한 변수만 선택하는 것이 좋습니다. ID, 날짜, 사후 변수는 제외하세요."
        )

        if target_type == "연속형":
            direction = st.radio("목표 방향", ["이하", "이상"], horizontal=True)
            default_target = float(base_pred[0]) * 0.9 if direction == "이하" else float(base_pred[0]) * 1.1
            target_value = st.number_input("목표 예측값", value=float(default_target))
            desired_rule = {"direction": direction, "target_value": target_value}
        else:
            y_values = sorted(pd.Series(st.session_state.y_train).astype(str).unique().tolist())
            target_class = st.selectbox("목표 클래스", y_values, index=len(y_values) - 1 if y_values else 0)
            min_prob = st.slider("목표 클래스 1 최소 확률", 0.0, 1.0, 0.5, 0.05)
            desired_rule = {"target_class": target_class, "min_prob": min_prob}

        c1, c2, c3 = st.columns(3)
        with c1:
            n_samples = st.number_input("탐색 후보 수", min_value=200, max_value=20000, value=3000, step=200)
        with c2:
            top_k = st.number_input("추천 대안 수", min_value=1, max_value=20, value=5, step=1)
        with c3:
            random_state = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)

        if st.button("Counterfactual XAI 실행", type="primary"):
            with st.spinner("목표 조건을 만족하는 대안을 탐색 중입니다..."):
                summary_df, cf_df = generate_counterfactual_xai(
                    input_df=input_df,
                    model=model,
                    preprocess_info=st.session_state.preprocess_info,
                    X_original=X_original,
                    target_type=target_type,
                    desired_rule=desired_rule,
                    mutable_features=mutable_features,
                    n_samples=int(n_samples),
                    random_state=int(random_state),
                    top_k=int(top_k)
                )

            if summary_df.empty:
                st.warning("목표 조건을 만족하는 Counterfactual 대안을 찾지 못했습니다. 목표를 완화하거나 변경 가능한 변수를 늘려보세요.")
            else:
                st.session_state.cf_xai_result = cf_df
                st.success("Counterfactual XAI 탐색 완료")
                st.write("추천 변경안 요약")
                st.dataframe(summary_df, width="stretch")

                st.write("Counterfactual 후보 원본")
                st.dataframe(cf_df, width="stretch")

                download_csv_button(summary_df, "Counterfactual 요약 저장", "counterfactual_xai_summary.csv")
                download_csv_button(cf_df, "Counterfactual 후보 전체 저장", "counterfactual_xai_candidates.csv")

                best = summary_df.iloc[0]
                st.info(f"가장 가까운 대안은 변경개수 {int(best['변경개수'])}개, 거리점수 {best['거리점수']:.4f}입니다.")


# ============================================================
# 16. 신규 데이터 예측
# ============================================================
elif step == "16. 신규 데이터 예측":
    st.header("16. 신규 데이터 예측 및 성능평가")

    if st.session_state.best_model is None:
        st.warning("먼저 모델 학습을 실행하세요.")
        st.stop()

    new_file = st.file_uploader("신규 데이터 업로드 (CSV/XLSX)", type=["csv", "xlsx"])

    if new_file:
        new_df, enc = read_uploaded_file(new_file)
        st.write("인코딩:", enc)
        st.dataframe(new_df.head())

        target_col = st.session_state.target
        has_target = target_col in new_df.columns

        X_new = new_df.drop(columns=[target_col]) if has_target else new_df.copy()
        X_new_p = preprocess_new_data(X_new, st.session_state.preprocess_info)

        model = st.session_state.best_model
        pred = model.predict(X_new_p)

        result_df = new_df.copy()
        result_df["예측값"] = pred

        if st.session_state.target_type == "범주형" and hasattr(model, "predict_proba"):
            result_df["예측확률_1"] = model.predict_proba(X_new_p)[:, 1]

        st.subheader("신규 데이터 예측 결과")
        st.dataframe(result_df)

        if has_target:
            st.subheader("신규 데이터 성능평가")

            y_true = new_df[target_col]

            if st.session_state.target_type == "연속형":
                metric = regression_metrics(y_true, pred)
            else:
                prob = result_df["예측확률_1"].values if "예측확률_1" in result_df.columns else pred
                metric = classification_metrics(y_true, pred, prob)

            st.dataframe(pd.DataFrame([metric]))

        download_csv_button(result_df, "신규 데이터 예측 결과 저장", "new_data_prediction_result.csv")