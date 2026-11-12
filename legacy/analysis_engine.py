"""Data ingestion, profiling and leakage-aware model evaluation."""
import hashlib

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, StratifiedKFold, TimeSeriesSplit, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler, RobustScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from statistical_tests import hypothesis_test


LIMIT = 10**9
FORMATS = {".csv", ".xlsx", ".parquet"}


class RobustClipper(TransformerMixin, BaseEstimator):
    """Learn clipping limits exclusively from the training fold."""

    def __init__(self, method="iqr", factor=1.5):
        self.method = method
        self.factor = factor

    def fit(self, values, y=None):
        if self.factor <= 0:
            raise ValueError("이상치 계수는 양수여야 합니다.")
        if self.method == "iqr":
            first, third = np.quantile(values, [.25, .75], axis=0)
            radius = self.factor * (third-first)
            self.lower_, self.upper_ = first-radius, third+radius
        elif self.method == "mad":
            center = np.median(values, axis=0)
            radius = self.factor * 1.4826 * np.median(np.abs(values-center), axis=0)
            self.lower_, self.upper_ = center-radius, center+radius
        else:
            raise ValueError("이상치 방법은 iqr 또는 mad입니다.")
        constant = self.lower_ == self.upper_
        self.lower_[constant], self.upper_[constant] = -np.inf, np.inf
        return self

    def transform(self, values):
        return np.clip(values, self.lower_, self.upper_)


def read(path, options=None, preview_rows=None):
    options = options or {}
    suffix = path.suffix.lower()
    if suffix == ".csv":
        for encoding in ([options["encoding"]] if options.get("encoding") else ["utf-8-sig", "cp949"]):
            try:
                return pd.read_csv(path, encoding=encoding, sep=options.get("separator", ","),
                                   usecols=options.get("columns"), nrows=preview_rows, low_memory=True)
            except UnicodeDecodeError:
                continue
        raise ValueError("CSV 인코딩을 지정하세요.")
    if suffix == ".xlsx":
        return pd.read_excel(path, sheet_name=options.get("sheet", 0), usecols=options.get("columns"), nrows=preview_rows)
    if suffix == ".parquet":
        if preview_rows is not None:
            from pyarrow.parquet import ParquetFile
            with ParquetFile(path) as source:
                first_batch = next(source.iter_batches(batch_size=preview_rows, columns=options.get("columns")), None)
                return first_batch.to_pandas() if first_batch is not None else pd.DataFrame(columns=options.get("columns") or source.schema.names)
        return pd.read_parquet(path, columns=options.get("columns"))
    raise ValueError("지원 형식은 CSV, XLSX, Parquet입니다.")


def fingerprint(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(paths, config):
    if not paths or sum(p.stat().st_size for p in paths) > LIMIT:
        raise ValueError("실행당 원본 합계는 1GB 이하여야 합니다.")
    files = [{"name": p.name, "sha256": fingerprint(p), "bytes": p.stat().st_size} for p in paths]
    options = config.get("read", {})
    frame = read(paths[0], options.get(paths[0].name))
    audit = []
    keys = config.get("keys", [])
    for path in paths[1:]:
        right = read(path, options.get(path.name))
        if config.get("combine") == "append":
            if set(frame.columns) != set(right.columns):
                raise ValueError(f"행 추가 파일의 열이 일치하지 않습니다: {path.name}")
            before = len(frame)
            frame = pd.concat([frame, right], ignore_index=True)
            audit.append({"operation": f"행 추가 {path.name}", "before": before, "after": len(frame)})
            continue
        if not keys:
            raise ValueError("다중 파일에 복합 결합 키를 지정하세요.")
        left_key, right_key = frame[keys], right[keys]
        coverage = left_key.dropna().drop_duplicates().merge(
            right_key.dropna().drop_duplicates(), on=keys, how="outer", indicator=True
        )
        before = len(frame)
        how = config.get("join", "left")
        if how not in {"left", "inner", "outer"}:
            raise ValueError("결합 방식은 left, inner, outer입니다.")
        # Missing keys are kept only on sides retained by the requested join.
        left_valid = left_key.notna().all(axis=1)
        right_valid = right_key.notna().all(axis=1)
        cardinality = config.get("cardinality", "one_to_one")
        left_counts = frame.loc[left_valid].groupby(keys, observed=True).size().rename("left")
        right_counts = right.loc[right_valid].groupby(keys, observed=True).size().rename("right")
        counts = left_counts.to_frame().join(right_counts, how="outer").fillna(0)
        both = (counts.left > 0) & (counts.right > 0)
        estimated = int((counts.loc[both, "left"] * counts.loc[both, "right"]).sum())
        if how in {"left", "outer"}:
            estimated += int(counts.loc[~both, "left"].sum()) + int((~left_valid).sum())
        if how == "outer":
            estimated += int(counts.loc[~both, "right"].sum()) + int((~right_valid).sum())
        if estimated > int(config.get("max_join_rows", 5_000_000)):
            raise ValueError(f"예상 결합 행 수 {estimated:,}가 실행 한도를 초과합니다.")
        merged = frame.loc[left_valid].merge(
            right.loc[right_valid], on=keys, how=how, validate=cardinality, suffixes=("", f"_{path.stem}")
        )
        if how in {"left", "outer"} and not left_valid.all():
            empty = right.iloc[:0]
            merged = pd.concat([merged, frame.loc[~left_valid].merge(empty, on=keys, how="left", suffixes=("", f"_{path.stem}"))], ignore_index=True)
        if how == "outer" and not right_valid.all():
            merged = pd.concat([merged, frame.iloc[:0].merge(right.loc[~right_valid], on=keys, how="right", suffixes=("", f"_{path.stem}"))], ignore_index=True)
        audit.append({"operation": f"결합 {path.name}", "before": before, "after": len(merged),
                      "keys": coverage["_merge"].value_counts().to_dict(),
                      "left_missing": int((~left_valid).sum()), "right_missing": int((~right_valid).sum()),
                      "left_duplicate_rows": int(left_key.loc[left_valid].duplicated(keep=False).sum()),
                      "right_duplicate_rows": int(right_key.loc[right_valid].duplicated(keep=False).sum())})
        frame = merged
    for rule in config.get("rules", []):
        before = len(frame)
        column, operation = rule["column"], rule["operation"]
        values = frame[column]
        affected = 0
        if operation == "numeric":
            converted = pd.to_numeric(values, errors="coerce")
            affected = int((values.notna() & converted.isna()).sum())
            frame[column] = converted
        elif operation == "date":
            converted = pd.to_datetime(values, errors="coerce", format=rule.get("format"), utc=True)
            affected = int((values.notna() & converted.isna()).sum())
            frame[column] = converted
        elif operation == "unit":
            frame[column] = pd.to_numeric(values) * float(rule["factor"]) + float(rule.get("offset", 0))
            affected = int(values.notna().sum())
        elif operation in {"range", "allowed"}:
            if operation == "range":
                valid = pd.to_numeric(values).between(float(rule["min"]), float(rule["max"]))
            else:
                valid = values.isin(rule["values"])
            invalid = values.notna() & ~valid
            affected = int(invalid.sum())
            action = rule.get("action", "flag")
            if action == "exclude":
                frame = frame.loc[~invalid].copy()
            elif action == "mask":
                frame[column] = values.mask(invalid)
            elif action == "flag":
                frame[rule.get("output", f"{column}_invalid")] = invalid
            else:
                raise ValueError("범위 규칙 action은 flag, mask, exclude입니다.")
        elif operation == "drop_missing":
            frame = frame.loc[values.notna()].copy()
            affected = before - len(frame)
        elif operation == "deduplicate":
            frame = frame.drop_duplicates(rule.get("keys", [column]), keep=rule.get("keep", "first"))
            affected = before - len(frame)
        elif operation in {"ratio", "difference", "sum", "product", "log"}:
            numeric = pd.to_numeric(values)
            if operation == "log":
                transformed = np.log(numeric.where(numeric > 0))
            else:
                other = pd.to_numeric(frame[rule["other"]])
                if operation == "ratio":
                    transformed = numeric / other.where(other != 0)
                elif operation == "difference":
                    transformed = numeric - other
                elif operation == "sum":
                    transformed = numeric + other
                else:
                    transformed = numeric * other
            output = rule["output"]
            if output in frame:
                raise ValueError(f"파생 열이 이미 존재합니다: {output}")
            frame[output] = transformed
            affected = int(transformed.notna().sum())
        else:
            raise ValueError(f"알 수 없는 처리: {operation}")
        audit.append({"operation": operation, "column": column, "before": before, "after": len(frame), "affected": affected})
    if frame.empty:
        raise ValueError("전처리 후 관측치가 없습니다.")
    return frame.reset_index(drop=True), files, audit


def explore(frame, config):
    for column, values in config.get("filters", {}).items():
        frame = frame.loc[frame[column].isin(values)]
    sample_size = int(config.get("sample", 2500))
    if not 1 <= sample_size <= 20000:
        raise ValueError("차트 표본은 1~20,000행입니다.")
    seed = int(config.get("seed", 42))
    sample = frame.sample(min(sample_size, len(frame)), random_state=seed)
    columns = config.get("columns") or list(frame.columns[:30])
    summary = []
    for column in columns:
        values = frame[column]
        entry = {"column": column, "type": str(values.dtype), "valid": int(values.count()),
                 "missing": int(values.isna().sum()), "unique": int(values.nunique())}
        if pd.api.types.is_numeric_dtype(values):
            entry.update(values.describe(percentiles=[.05, .25, .5, .75, .95]).to_dict())
            entry["skew"] = values.skew()
        else:
            entry["top"] = {str(k): v for k, v in values.value_counts().head(10).items()}
        summary.append(entry)
    traces = []
    x, y, group = (config.get(key) for key in ("x", "y", "group"))
    chart = config.get("chart", "histogram")
    if not x:
        x = next(iter(frame[columns].select_dtypes("number").columns), None)
    if x:
        bins = int(config.get("bins", 20))
        if not 1 <= bins <= 200:
            raise ValueError("히스토그램 구간 수는 1~200입니다.")
        bin_edges = None
        if chart == "histogram":
            values = pd.to_numeric(sample[x], errors="coerce")
            bin_edges = np.histogram_bin_edges(values[np.isfinite(values)], bins=bins)
        groups = sample.groupby(group, dropna=False, observed=True) if group else [("전체", sample)]
        for label, subset in groups:
            if len(traces) >= 30:
                break
            if chart == "histogram":
                numeric = pd.to_numeric(subset[x], errors="coerce")
                counts, edges = np.histogram(numeric[np.isfinite(numeric)], bins=bin_edges)
                traces.append({"label": str(label), "x": ((edges[:-1] + edges[1:]) / 2).tolist(), "y": counts.tolist()})
            elif chart == "frequency":
                counts = subset[x].value_counts().head(int(config.get("top", 20)))
                traces.append({"label": str(label), "x": counts.index.astype(str).tolist(), "y": counts.tolist()})
            elif chart in {"scatter", "trend"} and y:
                points = subset[[x, y]].dropna()
                if chart == "trend":
                    points = points.sort_values(x)
                traces.append({"label": str(label), "x": points[x].tolist(), "y": points[y].tolist()})
            elif chart == "ecdf":
                ordered = np.sort(pd.to_numeric(subset[x], errors="coerce").dropna())
                traces.append({"label": str(label), "x": ordered.tolist(), "y": (np.arange(1, len(ordered)+1)/max(1, len(ordered))).tolist()})
    numeric = frame[columns].select_dtypes("number").iloc[:, :40]
    valid = numeric.notna().astype("int64")
    output = {"rows": len(frame), "chart_rows": len(sample), "seed": seed, "summary": summary,
              "correlation": numeric.corr(method=config.get("correlation", "spearman")).to_dict(),
              "pair_counts": (valid.T @ valid).to_dict(), "chart": {"type": chart, "series": traces, "x": x, "y": y}}
    if x and y and config.get("crosstab"):
        output["crosstab"] = pd.crosstab(frame[x], frame[y]).iloc[:30, :30].to_dict()
    if y and group:
        grouped = frame.groupby(group, observed=True)[y]
        if pd.api.types.is_numeric_dtype(frame[y]):
            output["group_summary"] = grouped.agg(["count", "mean", "median", "std"]).head(30).reset_index().to_dict("records")
    if config.get("test"):
        output["test"] = hypothesis_test(frame, config)
    return output


def split(frame, config, classification):
    mode = config.get("split", "random")
    seed, fraction = int(config.get("seed", 42)), float(config.get("test_size", .2))
    folds = int(config.get("folds", 5))
    if not 0 < fraction < 1 or not 2 <= folds <= 20:
        raise ValueError("테스트 비율은 0과 1 사이이며 폴드는 2~20입니다.")
    indices = np.arange(len(frame))
    groups = frame[config["group"]] if config.get("group") else None
    if mode in {"time", "group_time"}:
        times = pd.to_datetime(frame[config["time"]], errors="raise", utc=True)
        if times.isna().any():
            raise ValueError("시간 분할 열에 결측이 있습니다.")
        unique_times = np.sort(times.unique())
        gap = int(config.get("gap", 0))
        if gap < 0:
            raise ValueError("시간 간격은 0 이상이어야 합니다.")
        cut = int(len(unique_times) * (1 - fraction))
        if cut <= gap or cut >= len(unique_times):
            raise ValueError("시간 구간이 분할에 부족합니다.")
        train = indices[times.isin(unique_times[:cut-gap])]
        test = indices[times.isin(unique_times[cut:])]
        if mode == "group_time":
            if groups is None or groups.isna().any():
                raise ValueError("그룹·시간 분할에 결측 없는 그룹 열이 필요합니다.")
            train = train[~groups.iloc[train].isin(groups.iloc[test])]
        train_times = times.iloc[train].reset_index(drop=True)
        dates = np.sort(train_times.unique())
        cv = []
        for a, b in TimeSeriesSplit(n_splits=folds, gap=gap).split(dates):
            left = np.flatnonzero(train_times.isin(dates[a]))
            right = np.flatnonzero(train_times.isin(dates[b]))
            if mode == "group_time":
                left = left[~groups.iloc[train[left]].isin(groups.iloc[train[right]])]
            cv.append((left, right))
    elif mode == "group":
        if groups is None or groups.isna().any():
            raise ValueError("그룹 분할에는 결측 없는 그룹 열이 필요합니다.")
        train, test = next(GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed).split(frame, groups=groups))
        cv = list(GroupKFold(n_splits=folds).split(train, groups=groups.iloc[train]))
    elif mode in {"random", "stratified"}:
        stratify = frame[config["target"]] if classification and mode == "stratified" else None
        train, test = train_test_split(indices, test_size=fraction, random_state=seed, stratify=stratify)
        if stratify is not None:
            cv = list(StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(train, stratify.iloc[train]))
        else:
            cv = list(KFold(n_splits=folds, shuffle=True, random_state=seed).split(train))
    else:
        raise ValueError("알 수 없는 분할 방식입니다.")
    if not len(train) or not len(test) or any(not len(a) or not len(b) for a, b in cv):
        raise ValueError("분할 제약을 만족하는 학습·검증 관측치가 부족합니다.")
    return train, test, cv


def evaluate(actual, prediction, probability, classes, classification):
    if not classification:
        return {"mae": metrics.mean_absolute_error(actual, prediction),
                "rmse": metrics.root_mean_squared_error(actual, prediction),
                "bias": float(np.mean(prediction - actual)), "r2": metrics.r2_score(actual, prediction)}
    labels = np.unique(np.concatenate([np.asarray(actual), np.asarray(classes)]))
    result = {"accuracy": metrics.accuracy_score(actual, prediction),
              "f1_macro": metrics.f1_score(actual, prediction, labels=labels, average="macro", zero_division=0),
              "classes": metrics.classification_report(actual, prediction, labels=labels, zero_division=0, output_dict=True),
              "confusion_labels": labels.tolist(),
              "confusion": metrics.confusion_matrix(actual, prediction, labels=labels).tolist()}
    if len(classes) == 2 and len(labels) == 2 and len(np.unique(actual)) == 2:
        binary = np.asarray(actual) == classes[1]
        result.update({"roc_auc": metrics.roc_auc_score(binary, probability[:, 1]),
                       "average_precision": metrics.average_precision_score(binary, probability[:, 1]),
                       "brier": metrics.brier_score_loss(binary, probability[:, 1])})
    return result


def model(frame, config):
    task = config.get("task", "regression")
    if task not in {"regression", "classification"}:
        raise ValueError("과제는 regression 또는 classification입니다.")
    classification = task == "classification"
    target = config["target"]
    excluded = {target, config.get("time"), config.get("group"), *config.get("exclude", [])}
    features = list(dict.fromkeys(config.get("features", [])))
    if not features or excluded.intersection(features):
        raise ValueError("예측 시점에 확보되는 특징을 지정하고 목표·시간·그룹·제외 열을 제거하세요.")
    frame = frame.loc[frame[target].notna()].copy()
    frame[target] = frame[target].astype(str) if classification else pd.to_numeric(frame[target])
    if not classification and not np.isfinite(frame[target]).all():
        raise ValueError("회귀 목표에 무한값이 있습니다.")
    train, test, folds = split(frame, config, classification)
    data, actual = frame[features], frame[target]
    numeric = list(data.select_dtypes("number").columns)
    category = [column for column in features if column not in numeric]
    scale = config.get("scale", "standard")
    scaling = {"standard": StandardScaler(), "robust": RobustScaler(), "none": "passthrough"}[scale]
    steps = []
    if numeric:
        numeric_steps = [("impute", SimpleImputer(strategy=config.get("impute", "median"), keep_empty_features=True))]
        if config.get("clip"):
            numeric_steps.append(("clip", RobustClipper(config["clip"], float(config.get("clip_factor", 1.5)))))
        if config.get("power"):
            numeric_steps.append(("power", PowerTransformer(method=config["power"], standardize=False)))
        numeric_steps.append(("scale", scaling))
        steps.append(("numeric", Pipeline(numeric_steps), numeric))
    if category:
        steps.append(("category", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encode", OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=int(config.get("max_categories", 32)), sparse_output=False))
        ]), category))
    seed = int(config.get("seed", 42))
    trees = int(config.get("trees", 100))
    if not 10 <= trees <= 500:
        raise ValueError("트리 수는 10~500입니다.")
    weight = config.get("class_weight")
    candidates = (
        {"baseline": DummyClassifier(), "linear": LogisticRegression(max_iter=500, class_weight=weight),
         "tree": DecisionTreeClassifier(max_depth=int(config.get("depth", 5)), random_state=seed, class_weight=weight),
         "forest": RandomForestClassifier(n_estimators=trees, random_state=seed, class_weight=weight, n_jobs=1),
         "boosting": HistGradientBoostingClassifier(max_iter=trees, random_state=seed, class_weight=weight)}
        if classification else
        {"baseline": DummyRegressor(strategy="median"), "linear": Ridge(alpha=float(config.get("alpha", 1))),
         "tree": DecisionTreeRegressor(max_depth=int(config.get("depth", 5)), random_state=seed),
         "forest": RandomForestRegressor(n_estimators=trees, random_state=seed, n_jobs=1),
         "boosting": HistGradientBoostingRegressor(max_iter=trees, random_state=seed)}
    )
    scoring = config.get("scoring") or ("f1_macro" if classification else "neg_mean_absolute_error")
    results, pipelines = [], {}
    for name in dict.fromkeys(["baseline", *config.get("models", ["linear", "forest"])]):
        pipeline = Pipeline([("preprocess", ColumnTransformer(clone(steps, safe=False))), ("model", candidates[name])])
        try:
            run = cross_validate(pipeline, data.iloc[train], actual.iloc[train], cv=folds, scoring=scoring, error_score="raise")
            scores = run["test_score"]
            if not np.isfinite(scores).all():
                raise ValueError("유한한 검증 점수를 계산하지 못했습니다.")
            results.append({"model": name, "scores": scores.tolist(), "mean": scores.mean(),
                            "sd": scores.std(), "fit_seconds": run["fit_time"].sum()})
            pipelines[name] = pipeline
        except (ValueError, TypeError) as error:
            results.append({"model": name, "error": str(error)})
    ranked = [row for row in results if "mean" in row]
    if not ranked:
        raise ValueError(f"학습 가능한 모델이 없습니다: {results}")
    selected = max(ranked, key=lambda row: row["mean"])["model"]
    fitted = pipelines[selected].fit(data.iloc[train], actual.iloc[train])
    pred = fitted.predict(data.iloc[test])
    probabilities = fitted.predict_proba(data.iloc[test]) if classification else None
    classes = fitted.classes_ if classification else None
    segments = []
    for column in config.get("segments", []):
        values = frame.iloc[test][column].reset_index(drop=True)
        for key, ix in values.groupby(values, dropna=False, observed=True).indices.items():
            if len(segments) >= 100:
                break
            segments.append({"column": column, "value": key, "n": len(ix), "metrics": evaluate(
                actual.iloc[test].iloc[ix], pred[ix], probabilities[ix] if classification else None, classes, classification)})
    output = {"selected": selected, "cv_scoring": scoring, "candidates": results,
              "train_rows": len(train), "test_rows": len(test), "test": evaluate(actual.iloc[test], pred, probabilities, classes, classification),
              "segments": segments}
    if not classification:
        count = min(2500, len(test))
        plot = np.random.default_rng(seed).choice(len(test), count, replace=False)
        output["chart_data"] = {"type": "scatter", "x": "예측값", "y": "잔차", "series": [{
            "label": "최종 검증", "x": pred[plot].tolist(),
            "y": (actual.iloc[test].to_numpy()[plot] - pred[plot]).tolist()}]}
    predictions = pd.DataFrame({"row_id": frame.index[test], "actual": actual.iloc[test].to_numpy(), "prediction": pred})
    if classification:
        for index, label in enumerate(classes):
            predictions[f"probability_{label}"] = probabilities[:, index]
    reference = {}
    for column in features:
        values = data.iloc[train][column]
        spec = {"missing": float(values.isna().mean())}
        if column in numeric and values.notna().any():
            spec["quantiles"] = values.quantile([.05, .5, .95]).tolist()
            finite = values[np.isfinite(values)]
            if not finite.empty:
                boundaries = np.unique(finite.quantile(np.linspace(.1, .9, 9)))
                spec["bins"] = boundaries.tolist()
                spec["histogram"] = np.histogram(finite, bins=np.r_[-np.inf, boundaries, np.inf])[0].tolist()
        elif column in category:
            spec["top"] = values.astype("string").value_counts(normalize=True).head(20).to_dict()
        reference[column] = spec
    return output, predictions, {"pipeline": fitted, "features": features, "task": task, "target": target, "reference": reference}, frame.iloc[train]


def batch(frame, artifact, config):
    features = artifact["features"]
    missing = [column for column in features if column not in frame]
    if missing:
        raise ValueError(f"배치에 누락된 특징: {missing}")
    data = frame[features]
    pipeline = artifact["pipeline"]
    prediction = pipeline.predict(data)
    classification = artifact["task"] == "classification"
    probability = pipeline.predict_proba(data) if classification else None
    classes = pipeline.classes_ if classification else None
    predictions = pd.DataFrame({"row_id": frame.index, "prediction": prediction})
    if config.get("id"):
        predictions["record_id"] = frame[config["id"]]
    if classification:
        for index, label in enumerate(classes):
            predictions[f"probability_{label}"] = probability[:, index]
    drift = []
    drift_method = config.get("drift")
    if drift_method and drift_method not in {"psi", "js"}:
        raise ValueError("분포 변화 지표는 psi 또는 js입니다.")
    for column, baseline in artifact["reference"].items():
        values = data[column]
        row = {"column": column, "baseline": baseline, "missing": values.isna().mean(),
               "missing_change": values.isna().mean() - baseline["missing"]}
        if "quantiles" in baseline:
            row["quantiles"] = pd.to_numeric(values).quantile([.05, .5, .95]).tolist()
            row["outside_reference_90pct"] = ((values < baseline["quantiles"][0]) | (values > baseline["quantiles"][2])).mean()
            if drift_method and "histogram" in baseline:
                finite = values[np.isfinite(values)]
                if len(finite):
                    counts = np.histogram(finite, bins=np.r_[-np.inf, baseline["bins"], np.inf])[0]
                    reference = np.asarray(baseline["histogram"], dtype=float)
                    p = np.maximum(reference / reference.sum(), 1e-6)
                    q = np.maximum(counts / counts.sum(), 1e-6)
                    p, q = p/p.sum(), q/q.sum()
                    if drift_method == "psi":
                        row["psi"] = float(np.sum((q-p)*np.log(q/p)))
                    else:
                        from scipy.spatial.distance import jensenshannon
                        row["js_distance"] = float(jensenshannon(p, q, base=2))
                    row["drift_valid_n"] = len(finite)
            elif drift_method:
                row["drift_note"] = "분포 기준이 없는 기존 모델입니다. 새 기준을 학습하세요."
        if "top" in baseline:
            distribution = values.astype("string").value_counts(normalize=True)
            row["reference_top_share"] = {key: distribution.get(key, 0.0) for key in baseline["top"]}
            row["outside_reference_top"] = float(1 - sum(row["reference_top_share"].values())) if len(distribution) else None
        drift.append(row)
    output = {"rows": len(frame), "features": drift, "label_coverage": 0.0,
              "prediction_distribution": pd.Series(prediction).value_counts(normalize=True).to_dict() if classification else pd.Series(prediction).describe().to_dict()}
    target = artifact["target"]
    if target in frame:
        observed = frame[target].notna().to_numpy()
        output["label_coverage"] = float(observed.mean())
        if observed.any():
            actual = frame.loc[observed, target]
            actual = actual.astype(str) if classification else pd.to_numeric(actual)
            output["metrics"] = evaluate(actual, prediction[observed], probability[observed] if classification else None, classes, classification)
    return output, predictions
