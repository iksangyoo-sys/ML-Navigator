"""Statistical process control and designed-experiment workflows."""
from itertools import combinations, product
from math import ceil, gamma, sqrt

import numpy as np
import pandas as pd
from scipy import stats


def spc(frame, cfg):
    chart, value, time = cfg.get("chart", "imr"), cfg["value"], cfg["time"]
    data = frame.copy()
    data["_time"] = pd.to_datetime(data[time], utc=True, errors="raise")
    data["_value"] = pd.to_numeric(data[value])
    if data[["_time", "_value"]].isna().any().any() or not np.isfinite(data["_value"]).all():
        raise ValueError("SPC에 결측 또는 무한 관측값이 있습니다.")
    data = data.sort_values("_time", kind="stable")
    if chart == "imr":
        if data["_time"].duplicated().any():
            raise ValueError("I–MR에는 관측 순서가 유일해야 합니다.")
        rows = pd.DataFrame({"time": data["_time"], "stat": data["_value"], "n": 1})
    else:
        group = cfg["group"]
        if data[group].isna().any():
            raise ValueError("부분군 식별자에 결측이 있습니다.")
        grouped = data.groupby(group, observed=True, sort=False)
        rows = grouped.agg(time=("_time", "max"), n=("_value", "size"), total=("_value", "sum"), stat=("_value", "mean"), sd=("_value", "std"), spread=("_value", lambda x: x.max()-x.min()))
        rows = rows.sort_values("time")
        if chart in {"p", "u"}:
            if (data["_value"] < 0).any() or (data["_value"] % 1 != 0).any():
                raise ValueError("불량·결점 수는 음이 아닌 정수여야 합니다.")
            if cfg.get("denominator"):
                denom = pd.to_numeric(data[cfg["denominator"]])
                if denom.isna().any() or (denom <= 0).any() or not np.isfinite(denom).all():
                    raise ValueError("검사 수 또는 노출량은 양수여야 합니다.")
                if chart == "p" and ((denom % 1 != 0).any() or (data["_value"] > denom).any()):
                    raise ValueError("p 관리도는 정수 검사 수와 검사 수 이하 불량 건수가 필요합니다.")
                rows["n"] = denom.groupby(data[group], observed=True).sum()
            elif chart == "p" and (data["_value"] > 1).any():
                raise ValueError("개별 불량 값은 0 또는 1이어야 합니다.")
            rows["stat"] = rows["total"] / rows["n"]
    start, end = (pd.Timestamp(cfg[key], tz="UTC") if len(cfg[key]) == 10 else pd.to_datetime(cfg[key], utc=True)
                  for key in ("start", "end"))
    if len(cfg["end"]) == 10:
        end += pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    mask = rows["time"].between(start, end)
    if chart != "imr":
        minimum_time = grouped["_time"].min().reindex(rows.index)
        crossing = ((minimum_time < start) & (rows["time"] >= start)) | ((minimum_time <= end) & (rows["time"] > end))
        if crossing.any():
            raise ValueError("기준 기간 경계에 걸친 부분군이 있습니다.")
    base = rows.loc[mask]
    if len(base) < 2:
        raise ValueError("기준 기간에 관측치 또는 부분군이 2개 이상 필요합니다.")
    center = float(base["stat"].mean())
    sigma = None
    auxiliary = None
    if chart == "imr":
        moving = rows["stat"].diff().abs()
        reference_mr = base["stat"].diff().abs().mean()
        sigma = reference_mr / 1.128
        auxiliary = {"name": "MR", "center": reference_mr, "lcl": 0.0, "ucl": 3.267 * reference_mr, "values": moving.tolist()}
        width = 3 * sigma
    elif chart in {"xbar_r", "xbar_s"}:
        if rows["n"].nunique() != 1 or rows["n"].iloc[0] < 2:
            raise ValueError("X̄–R/S는 동일한 크기의 부분군이 필요합니다.")
        n = int(rows["n"].iloc[0])
        if chart == "xbar_r":
            constants = {2: (1.128, 0, 3.267), 3: (1.693, 0, 2.574), 4: (2.059, 0, 2.282),
                         5: (2.326, 0, 2.114), 6: (2.534, 0, 2.004), 7: (2.704, .076, 1.924),
                         8: (2.847, .136, 1.864), 9: (2.970, .184, 1.816), 10: (3.078, .223, 1.777)}
            if n not in constants:
                raise ValueError("X̄–R의 부분군 크기는 2~10입니다.")
            d2, lo, hi = constants[n]
            mean_spread = base["spread"].mean()
            sigma = mean_spread / d2
            auxiliary = {"name": "R", "center": mean_spread, "lcl": lo*mean_spread, "ucl": hi*mean_spread, "values": rows["spread"].tolist()}
        else:
            from scipy.special import gammaln
            c4 = sqrt(2/(n-1)) * np.exp(gammaln(n/2) - gammaln((n-1)/2))
            mean_sd = base["sd"].mean()
            sigma = mean_sd / c4
            delta = 3 * sqrt(max(0, 1-c4*c4)) * sigma
            auxiliary = {"name": "S", "center": mean_sd, "lcl": max(0, mean_sd-delta), "ucl": mean_sd+delta, "values": rows["sd"].tolist()}
        width = 3 * sigma / sqrt(n)
    elif chart in {"p", "u"}:
        center = float(base["total"].sum() / base["n"].sum())
        variance = center * (1-center) if chart == "p" else center
        width = 3 * np.sqrt(variance / rows["n"])
    else:
        raise ValueError("관리도는 imr, xbar_r, xbar_s, p, u를 지원합니다.")
    rows["lcl"] = center-width
    rows["ucl"] = center+width
    if chart in {"p", "u"}:
        rows["lcl"] = rows["lcl"].clip(lower=0)
    if chart == "p":
        rows["ucl"] = rows["ucl"].clip(upper=1)
    rows["phase"] = np.where(mask, "I", np.where(rows["time"] > end, "II", "before"))
    rows["signal"] = (rows["stat"] > rows["ucl"]) | (rows["stat"] < rows["lcl"])
    if cfg.get("run_rule"):
        side = np.sign(rows["stat"] - center)
        rows["run_signal"] = side.rolling(8).sum().abs().eq(8)
        rows["signal"] |= rows["run_signal"]
    if auxiliary:
        values = np.asarray(auxiliary["values"])
        rows["spread_signal"] = (values < auxiliary["lcl"]) | (values > auxiliary["ucl"])
        rows["signal"] |= rows["spread_signal"]
    output = {"chart": chart, "center": center, "reference": {"start": start, "end": end, "groups": len(base)},
              "signals": int(rows["signal"].sum()), "auxiliary": auxiliary}
    if sigma is not None and "lsl" in cfg and "usl" in cfg:
        lower, upper = float(cfg["lsl"]), float(cfg["usl"])
        if lower >= upper:
            raise ValueError("규격 하한은 상한보다 작아야 합니다.")
        selected = data["_time"].between(start, end)
        overall = data.loc[selected, "_value"].std()
        output["capability"] = {"lsl": lower, "usl": upper, "n": int(selected.sum()),
                                "phase_i_signal_count": int(rows.loc[mask, "signal"].sum())}
        for sd, potential, actual in [(sigma, "cp", "cpk"), (overall, "pp", "ppk")]:
            if sd > 0:
                output["capability"].update({potential: (upper-lower)/(6*sd), actual: min(upper-center, center-lower)/(3*sd)})
    rows = rows.reset_index(names="subgroup")
    output["chart_data"] = {"type": "trend", "x": "실행 순서", "y": value, "series": [
        {"label": label, "x": list(range(1, len(rows)+1)), "y": series}
        for label, series in [("통계량", rows["stat"].tolist()), ("중심선", [center]*len(rows)),
                              ("UCL", rows["ucl"].tolist()), ("LCL", rows["lcl"].tolist())]]}
    return output, rows


def design(cfg):
    factors = cfg["factors"]
    names = [item["name"] for item in factors]
    n = len(names)
    if not 2 <= n <= 10 or len(set(names)) != n or any(not name or name.startswith("coded:") or name in {"run_id", "block", "repeat", "standard_order", "run_order"} for name in names):
        raise ValueError("서로 다른 유효한 요인 2~10개를 지정하세요.")
    bounds = np.array([[float(f["low"]), float(f["high"])] for f in factors])
    if not np.isfinite(bounds).all() or (bounds[:, 0] >= bounds[:, 1]).any():
        raise ValueError("요인 저수준과 고수준을 확인하세요.")
    method = cfg.get("design", "full")
    center = int(cfg.get("center", 0))
    repeat = int(cfg.get("repeat", 1))
    blocks = int(cfg.get("blocks", 1))
    budget = int(cfg.get("budget", 1000))
    if not 0 <= center <= 1000 or not 1 <= repeat <= 100 or not 1 <= blocks <= 100 or not 1 <= budget <= 10000:
        raise ValueError("중심점, 반복, 블록 및 실행 예산을 확인하세요.")
    generators = cfg.get("generators", {}) if method == "fractional" else {}
    if method == "fractional" and not generators:
        raise ValueError("부분 요인에는 생성식 매핑이 필요합니다.")
    independent = [name for name in names if name not in generators]
    if any(key not in names or not isinstance(term, list) or len(set(term)) < 2 or len(term) != len(set(term)) or not set(term) <= set(independent) for key, term in generators.items()):
        raise ValueError("생성식에는 서로 다른 독립 요인 2개 이상을 사용하세요.")
    points = []
    if method in {"full", "fractional", "ccd"}:
        for row in product([-1, 1], repeat=len(independent)):
            values = dict(zip(independent, row))
            values.update({key: np.prod([values[term] for term in terms]) for key, terms in generators.items()})
            points.append([values[name] for name in names])
        if method == "ccd":
            alpha = float(cfg.get("alpha", 1))
            if not np.isfinite(alpha) or alpha <= 0:
                raise ValueError("축점 alpha는 양수입니다.")
            for i in range(n):
                for sign in [-1, 1]:
                    point = [0.0]*n
                    point[i] = sign*alpha
                    points.append(point)
    elif method == "box_behnken" and n >= 3:
        for i, j in combinations(range(n), 2):
            for a, b in product([-1, 1], repeat=2):
                point = [0]*n
                point[i], point[j] = a, b
                points.append(point)
    else:
        raise ValueError("설계 유형 또는 요인 수를 확인하세요.")
    total = (len(points) + center) * repeat
    if total > budget or blocks > total:
        raise ValueError(f"실행 수 {total}와 블록 수가 예산 또는 설계 조건을 초과합니다.")
    coded = np.tile(np.asarray(points + [[0]*n]*center), (repeat, 1))
    actual = bounds.mean(axis=1) + coded * ((bounds[:, 1]-bounds[:, 0])/2)
    for i, factor in enumerate(factors):
        if np.any(actual[:, i] < float(factor.get("min", bounds[i, 0]))) or np.any(actual[:, i] > float(factor.get("max", bounds[i, 1]))):
            raise ValueError(f"요인 {names[i]}의 계획 조건이 허용 범위를 벗어났습니다.")
    plan = pd.DataFrame(actual, columns=names)
    for i, name in enumerate(names):
        plan[f"coded:{name}"] = coded[:, i]
    plan["repeat"] = np.repeat(np.arange(1, repeat+1), total//repeat)
    plan["standard_order"] = np.arange(1, total+1)
    plan["run_id"] = [f"R{i:05d}" for i in plan["standard_order"]]
    for rule in cfg.get("forbidden", []):
        if not rule or not set(rule) <= set(names):
            raise ValueError("금지 조합에 유효한 요인을 지정하세요.")
        forbidden = np.logical_and.reduce([np.isclose(plan[name], value) for name, value in rule.items()])
        plan = plan.loc[~forbidden].copy()
    if len(plan) < blocks:
        raise ValueError("금지 조합 제거 후 실행 수가 부족합니다.")
    rng = np.random.default_rng(int(cfg.get("seed", 42)))
    ordering = rng.permutation(len(plan))
    plan["block"] = np.argsort(ordering) % blocks + 1
    plan = plan.iloc[ordering].sort_values("block", kind="stable").reset_index(drop=True)
    plan["run_order"] = np.arange(1, len(plan)+1)
    matrix = design_matrix(plan, names, method in {"ccd", "box_behnken"})
    rank = np.linalg.matrix_rank(matrix)
    defining = []
    words = [{key, *terms} for key, terms in generators.items()]
    for length in range(1, len(words)+1):
        for selection in combinations(words, length):
            combined = set()
            for word in selection:
                combined ^= word
            defining.append(combined)
    aliases = {name: [sorted({name} ^ word) for word in defining] for name in names}
    return {"design": method, "factors": names, "planned": len(plan), "excluded": total-len(plan),
            "rank": rank, "terms": len(matrix.columns), "residual_df": len(plan)-rank,
            "resolution": min(map(len, defining)) if defining else None,
            "defining_words": [sorted(word) for word in defining], "main_effect_aliases": aliases}, plan


def design_matrix(frame, names, quadratic, terms=None):
    matrix = pd.DataFrame({"intercept": np.ones(len(frame))}, index=frame.index)
    for name in names:
        matrix[name] = frame[f"coded:{name}"]
    for a, b in combinations(names, 2):
        matrix[f"{a} × {b}"] = matrix[a]*matrix[b]
    if quadratic:
        for name in names:
            matrix[f"{name}²"] = matrix[name]**2
    if terms is not None:
        matrix = matrix[["intercept", *terms]]
    for block in sorted(frame["block"].unique())[1:]:
        matrix[f"block {block}"] = (frame["block"] == block).astype(float)
    return matrix


def doe_results(plan, response, design_cfg, cfg):
    name = cfg["response"]
    if response["run_id"].duplicated().any() or response["run_id"].isna().any():
        raise ValueError("실행 ID가 비어 있거나 중복되었습니다.")
    unknown = set(response["run_id"]) - set(plan["run_id"])
    if unknown:
        raise ValueError(f"설계에 없는 실행 ID: {sorted(unknown)[:10]}")
    data = plan.merge(response, on="run_id", how="left", validate="one_to_one", suffixes=("", "_observed"))
    if name in plan:
        name += "_observed"
    used = data.loc[data[name].notna()].copy()
    if used.empty:
        raise ValueError("분석할 반응값이 없습니다.")
    factors = [factor["name"] for factor in design_cfg["factors"]]
    for factor in design_cfg["factors"]:
        column = factor["name"]
        observed = f"{column}_observed"
        if observed in used:
            low, high = float(factor["low"]), float(factor["high"])
            used[f"coded:{column}"] = (pd.to_numeric(used[observed]).fillna(used[column]) - (low+high)/2) / ((high-low)/2)
    quadratic = design_cfg.get("design") in {"ccd", "box_behnken"}
    effects = cfg.get("effects")
    if effects == "quadratic" and not quadratic:
        raise ValueError("2차 모형에는 중심합성 또는 Box–Behnken 설계를 사용하세요.")
    terms = cfg.get("terms")
    if terms is None and effects == "main":
        terms = factors
    matrix = design_matrix(used, factors, quadratic and effects != "interaction", terms)
    x, y = matrix.to_numpy(float), pd.to_numeric(used[name]).to_numpy(float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("반응과 실제 요인에 유한값이 필요합니다.")
    coefficients, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    if rank < x.shape[1]:
        raise ValueError("선택 항의 교락 또는 누락으로 추정할 수 없습니다. terms에서 항을 선택하세요.")
    residual = y-x@coefficients
    sse = float(residual @ residual)
    dof = len(y)-rank
    estimates = []
    covariance_factor = np.linalg.pinv(x.T @ x)
    for i, term in enumerate(matrix.columns):
        beta = coefficients[i]
        se = sqrt(max(0, covariance_factor[i, i] * sse/dof)) if dof else None
        t_value = beta/se if se else None
        margin = stats.t.ppf(.975, dof)*se if se is not None else None
        estimates.append({"term": term, "coefficient": beta, "se": se, "p": 2*stats.t.sf(abs(t_value), dof) if t_value is not None else None,
                          "lower": beta-margin if margin is not None else None, "upper": beta+margin if margin is not None else None})
    settings = used[[f"coded:{factor}" for factor in factors] + ["block"]].copy()
    settings["_response"] = y
    means = settings.groupby(list(settings.columns[:-1]), dropna=False)["_response"].transform("mean")
    pure_df = len(settings) - len(settings.drop_duplicates(list(settings.columns[:-1])))
    pure_ss = float(np.sum((y-means)**2))
    lof_df = dof-pure_df
    lack_ss = max(0, sse-pure_ss)
    total_ss = float(np.sum((y-y.mean())**2))
    model_ss = max(0, total_ss-sse)
    f_value = (model_ss/(rank-1))/(sse/dof) if rank > 1 and dof > 0 and sse > 0 else None
    lof_f = (lack_ss/lof_df)/(pure_ss/pure_df) if pure_df > 0 and lof_df > 0 and pure_ss > 0 else None
    output = {"planned": len(plan), "observed": len(used), "missing_runs": data.loc[data[name].isna(), "run_id"].tolist(),
              "coefficients": estimates, "anova": {"model_ss": model_ss, "model_df": rank-1, "residual_ss": sse, "residual_df": dof,
              "f": f_value, "p": stats.f.sf(f_value, rank-1, dof) if f_value is not None else None,
              "pure_error_ss": pure_ss, "pure_error_df": pure_df, "lack_of_fit_ss": lack_ss, "lack_of_fit_df": lof_df,
              "lack_of_fit_f": lof_f, "lack_of_fit_p": stats.f.sf(lof_f, lof_df, pure_df) if lof_f is not None else None},
              "chart_data": {"type": "scatter", "x": "적합값", "y": "잔차", "series": [
                  {"label": "실험", "x": (x@coefficients).tolist(), "y": residual.tolist()}]}}
    confirmation = cfg.get("confirmation")
    if confirmation:
        baseline = np.asarray(confirmation["baseline"], dtype=float)
        candidate = np.asarray(confirmation["candidate"], dtype=float)
        direction = confirmation.get("direction", "increase")
        if direction not in {"increase", "decrease"} or min(len(baseline), len(candidate)) < 2 or not np.isfinite(baseline).all() or not np.isfinite(candidate).all():
            raise ValueError("확인실험은 유한한 독립 반복을 조건별 두 개 이상 입력하세요.")
        test = stats.ttest_ind(candidate, baseline, equal_var=False)
        ci = test.confidence_interval()
        improvement = candidate.mean()-baseline.mean()
        lower = ci.low if direction == "increase" else -ci.high
        threshold = float(confirmation["minimum"])
        output["confirmation"] = {"direction": direction, "minimum": threshold, "n_baseline": len(baseline), "n_candidate": len(candidate),
                                  "difference": improvement, "ci_95": [ci.low, ci.high], "pass": bool(lower >= threshold)}
    return output, data
