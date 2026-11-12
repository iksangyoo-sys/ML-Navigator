"""Statistical tests with explicit assumptions and effect estimates."""
import numpy as np
import pandas as pd
from scipy import stats


def hypothesis_test(frame, config):
    method = config["test"]
    x, y, group = (config.get(field) for field in ("x", "y", "group"))
    alpha = float(config.get("alpha_test", .05))
    alternative = config.get("alternative", "two-sided")
    if not 0 < alpha < 1 or alternative not in {"two-sided", "less", "greater"}:
        raise ValueError("유의수준과 대립가설 방향을 확인하세요.")

    def pair(numeric=True):
        if not x or not y or x == y:
            raise ValueError("서로 다른 X와 Y를 선택하세요.")
        data = frame[[x, y]].dropna()
        if numeric:
            data = data.apply(pd.to_numeric)
            data = data.loc[np.isfinite(data).all(axis=1)]
        if len(data) < 3:
            raise ValueError("유효한 관측 쌍이 3개 이상 필요합니다.")
        return data[x], data[y]

    def interval(test):
        ci = test.confidence_interval(confidence_level=1-alpha)
        return [ci.low, ci.high]

    effect, diagnostics = {}, {}
    if method in {"pearson", "spearman", "paired_t", "wilcoxon"}:
        a, b = pair()
        if method in {"pearson", "spearman"}:
            if a.nunique() < 2 or b.nunique() < 2:
                raise ValueError("상수 열에서는 상관을 계산할 수 없습니다.")
            test = (stats.pearsonr if method == "pearson" else stats.spearmanr)(a, b, alternative=alternative)
            effect = {"correlation": test.statistic}
            if method == "pearson":
                effect["confidence_interval"] = interval(test)
            assumptions = "관측쌍 간 독립. Pearson은 선형 관계, Spearman은 단조 관계를 평가합니다."
            if method == "spearman" and len(a) < 500:
                diagnostics["p_value_note"] = "점근 p값입니다. 소표본에서는 순열 검정을 검토하세요."
        elif method == "paired_t":
            test = stats.ttest_rel(a, b, alternative=alternative)
            effect = {"mean_x_minus_y": (a-b).mean(), "confidence_interval": interval(test)}
            assumptions = "한 행은 동일 실험 단위의 관측쌍입니다. 쌍은 독립이고 차이의 정규성을 검토합니다."
        else:
            if (a == b).all():
                raise ValueError("모든 관측쌍의 차이가 0입니다.")
            test = stats.wilcoxon(a-b, alternative=alternative)
            effect = {"median_paired_difference": np.median(a-b)}
            assumptions = "관측쌍은 독립이며 차이의 분포가 대칭이어야 합니다."
        diagnostics["n_pairs"] = len(a)
    elif method in {"chisquare", "fisher"}:
        a, b = pair(numeric=False)
        if min(a.nunique(), b.nunique()) < 2 or max(a.nunique(), b.nunique()) > 30:
            raise ValueError("범주형 X와 Y 각각에 2~30개의 관측 범주가 필요합니다.")
        table = pd.crosstab(a, b)
        if method == "fisher":
            if table.shape != (2, 2):
                raise ValueError("Fisher 검정에는 2×2 교차표가 필요합니다.")
            test = stats.fisher_exact(table, alternative=alternative)
            effect = {"odds_ratio": test.statistic}
        else:
            test = stats.chi2_contingency(table, correction=False)
            if (test.expected_freq < 5).any():
                raise ValueError("기대빈도 5 미만 셀이 있습니다. 2×2이면 Fisher 검정을 선택하세요.")
            effect = {"cramers_v": np.sqrt(test.statistic / (len(a)*min(table.shape[0]-1, table.shape[1]-1)))}
            diagnostics["minimum_expected_count"] = test.expected_freq.min()
        diagnostics["n"] = len(a)
        assumptions = "독립 관측의 도수입니다. 순서형 효과 방향과 범주 코딩은 별도로 해석합니다."
    else:
        if not group or not y or group == y:
            raise ValueError("범주형 그룹과 수치형 Y를 선택하세요.")
        values = pd.to_numeric(frame[y])
        valid = frame[group].notna() & np.isfinite(values)
        grouped = values.loc[valid].groupby(frame.loc[valid, group], observed=True, sort=False)
        sizes = grouped.size()
        if len(sizes) < 2 or len(sizes) > 30 or (sizes < 2).any():
            raise ValueError("그룹 2~30개와 그룹별 유효 관측치 2개 이상이 필요합니다.")
        groups = [part.to_numpy() for _, part in grouped]
        diagnostics["group_sizes_in_input_order"] = sizes.tolist()
        diagnostics["group_order"] = [str(label) for label in sizes.index]
        if method in {"welch", "mannwhitney"}:
            if len(groups) != 2:
                raise ValueError("이 검정에는 독립 그룹이 정확히 두 개 필요합니다.")
            a, b = groups
            if method == "welch":
                test = stats.ttest_ind(a, b, equal_var=False, alternative=alternative)
                effect = {"mean_first_minus_second": a.mean()-b.mean(), "confidence_interval": interval(test)}
            else:
                test = stats.mannwhitneyu(a, b, alternative=alternative)
                effect = {"rank_biserial": 2*test.statistic/(len(a)*len(b))-1}
        elif method == "welch_anova":
            if len(groups) < 3 or any(np.var(sample, ddof=1) == 0 for sample in groups):
                raise ValueError("Welch ANOVA에는 분산이 양수인 독립 그룹이 3개 이상 필요합니다.")
            test = stats.f_oneway(*groups, equal_var=False)
            effect = {"group_means_in_input_order": [np.mean(sample) for sample in groups]}
        elif method == "kruskal":
            test = stats.kruskal(*groups)
            denominator = sum(map(len, groups))-len(groups)
            effect = {"epsilon_squared": max(0, (test.statistic-len(groups)+1)/denominator)}
            if min(sizes) < 5:
                diagnostics["p_value_note"] = "5개 미만인 그룹이 있어 점근 p값에 주의하세요."
        else:
            raise ValueError("지원하는 통계 검정을 선택하세요.")
        assumptions = ("독립 관측과 그룹 내 평균 추론에 적합한 분포를 검토합니다."
                       if method in {"welch", "welch_anova"}
                       else "독립 관측의 순위를 비교합니다. 위치 차이로 해석하려면 분포 형태를 검토합니다.")
    if method == "fisher" and np.isinf(test.statistic):
        diagnostics["odds_ratio_note"] = "영 도수 셀로 오즈비가 무한대입니다."
    elif not np.isfinite(test.statistic):
        raise ValueError("상수 또는 퇴화 데이터로 유한한 검정 결과를 계산할 수 없습니다.")
    if not np.isfinite(test.pvalue):
        raise ValueError("상수 또는 퇴화 데이터로 유한한 검정 결과를 계산할 수 없습니다.")
    return {"method": method, "statistic": test.statistic, "p_value": test.pvalue, "alpha": alpha,
            "reject_null": bool(test.pvalue < alpha), "effect": effect, "diagnostics": diagnostics,
            "assumptions": assumptions}
