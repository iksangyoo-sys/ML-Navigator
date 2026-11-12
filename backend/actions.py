"""단계별 실행 액션. 각 액션은 (결과 키, 블록 목록)을 돌려주며 백그라운드 작업으로 실행된다."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .analysis import (blocks as B, causal as C, counterfactual as CF, dataio, eda, evaluate as EV, features as F,
                       merge as MG, missing as MS, modeling as MD, newdata as ND, outliers as OL, predict as PR,
                       recipe as R, target as T)
from .storage import lock
from .workspace import WS

REGISTRY: dict[str, dict] = {}


def action(name: str, mutating: bool = False):
    def deco(fn):
        REGISTRY[name] = {"fn": fn, "mutating": mutating}
        return fn
    return deco


def need(cond, msg):
    if not cond:
        raise ValueError(msg)


# ====================================================== 1단계
@action("s1_profile")
def s1_profile(ws: WS, p: dict, job):
    srcs = ws.st["sources"]
    need(srcs, "불러온 파일이 없습니다. 파일을 업로드하거나 data 폴더에서 선택하세요.")
    tabs = []
    for i, s in enumerate(srcs):
        job.progress(i / len(srcs), f"{s['name']} 프로파일 중")
        df = ws.source_df(s["id"])
        blocks, _ = dataio.profile_blocks(df, s["name"], s)
        tabs.append((s["name"], blocks))
    ws.merged()  # 결합 데이터가 없으면 첫 파일로 설정
    out = [B.tabs(tabs)] if len(tabs) > 1 else tabs[0][1]
    if len(srcs) > 1:
        out.insert(0, B.note(f"파일 {len(srcs)}개를 불러왔습니다. 2단계에서 결합 방식을 정하세요. 결합 전까지는 첫 파일을 작업 데이터로 사용합니다.", "info"))
    return "s1", out


# ====================================================== 2단계
@action("s2_keys")
def s2_keys(ws: WS, p: dict, job):
    named = ws.sources_named(p.get("sources"))
    need(len(named) >= 2, "결합하려면 파일이 2개 이상 필요합니다.")
    cand = MG.key_candidates(named)
    return "s2_keys", [B.table(pd.DataFrame(cand), "결합 키 후보 진단"),
                       B.interpret("왼쪽(첫 번째) 데이터 기준으로 공통 열의 고유비율과 매칭률을 계산했습니다. 고유비율이 1에 가까운 열은 식별자(키) 후보이고, "
                                   "매칭률이 낮으면 결합 후 미매칭 행이 많이 생깁니다. 여러 열을 함께 써야 고유해지는 복합키도 있습니다.")]


@action("s2_merge", mutating=True)
def s2_merge(ws: WS, p: dict, job):
    mode = p.get("mode", "rows")
    ids = p.get("sources") or [s["id"] for s in ws.st["sources"]]
    named = ws.sources_named(ids)
    need(named, "결합할 데이터를 선택하세요.")
    if mode == "single" or len(named) == 1:
        merged, blocks = named[0][1].copy(), [B.note(f"`{named[0][0]}` 한 파일을 작업 데이터로 사용합니다.", "success")]
    elif mode == "rows":
        merged, blocks = MG.concat_rows(named)
    else:
        merged, blocks = MG.merge_columns(named, p.get("keys") or [], p.get("how", "inner"))
    warns = ws.set_merged(merged, {"mode": mode, "sources": ids, "keys": p.get("keys"), "how": p.get("how")}, step=2)
    out = [B.note("데이터 결합 완료", "success")] + blocks + [
        B.metrics([("결합 후 행", len(merged)), ("결합 후 열", merged.shape[1]), ("결측 셀", int(merged.isna().sum().sum()))]),
        B.table(merged.head(10), "결합 데이터 (상위 10행)")]
    if warns:
        out.append(B.note("기존 레시피 중 적용되지 않은 연산:\n" + "\n".join(warns), "warn"))
    return "s2", out


# ====================================================== 3단계
def recipe_table(ws: WS):
    rec = ws.st["recipe"]
    if not rec:
        return B.note("아직 생성한 파생변수나 행 처리가 없습니다.", "info")
    return B.table(pd.DataFrame([{"순서": i + 1, "연산": R.describe(o)} for i, o in enumerate(rec)]), "현재 레시피 (신규 데이터에 그대로 재적용)")


@action("s3_derive", mutating=True)
def s3_derive(ws: WS, p: dict, job):
    op = dict(p["op"])
    need(op.get("op") in F.OPS, "파생 방식을 선택하세요.")
    df = ws.working()
    if op["op"] not in ("dummy", "date") and not op.get("new"):
        raise ValueError("파생변수명을 입력하세요.")
    if op.get("new") and op["new"] in df.columns and op["op"] not in ("dummy",):
        raise ValueError(f"`{op['new']}` 열이 이미 있습니다. 다른 이름을 쓰세요.")
    op = F.fit_op(df, op)
    after = F.apply_op(df, op)
    new_cols = [c for c in after.columns if c not in df.columns]
    st = ws.st
    st_recipe = st["recipe"] + [op]
    ws.update(recipe=st_recipe)
    warns = ws.rebuild(3, f"파생변수: {F.describe(op)}")
    show = [c for c in [op.get("source"), op.get("left"), op.get("right"), op.get("group")] if c and c in after.columns] + new_cols
    out = [B.note(f"{F.describe(op)} → 새 열 {len(new_cols)}개: {', '.join(new_cols[:20])}", "success"),
           B.table(after[show].head(10), "생성 결과 미리보기")]
    num_new = [c for c in new_cols if pd.api.types.is_numeric_dtype(after[c])]
    if num_new:
        out.append(B.table(after[num_new].describe().T.reset_index().rename(columns={"index": "변수"}), "새 변수 기술통계"))
    out.append(recipe_table(ws))
    if warns:
        out.append(B.note("\n".join(warns), "warn"))
    return "s3", out


@action("s3_remove", mutating=True)
def s3_remove(ws: WS, p: dict, job):
    rec = ws.st["recipe"]
    i = int(p["index"])
    need(0 <= i < len(rec), "잘못된 레시피 번호입니다.")
    removed = rec.pop(i)
    ws.update(recipe=rec)
    warns = ws.rebuild(3, f"레시피 삭제: {R.describe(removed)}")
    out = [B.note(f"삭제: {R.describe(removed)}", "success"), recipe_table(ws)]
    if warns:
        out.append(B.note("\n".join(warns), "warn"))
    return "s3", out


@action("s3_tukey")
def s3_tukey(ws: WS, p: dict, job):
    return "s3_tukey", F.tukey_ladder(ws.working(), p["column"])


# ====================================================== 4단계
@action("s4_target", mutating=True)
def s4_target(ws: WS, p: dict, job):
    df = ws.working()
    tgt = p.get("target")
    need(tgt in df.columns, "종속변수를 선택하세요.")
    ttype = p.get("ttype") or T.guess_type(df[tgt])
    if ttype == "classification":
        ttype = "binary" if df[tgt].nunique() == 2 else "multiclass"
    if ttype == "regression":
        need(pd.api.types.is_numeric_dtype(df[tgt]) or pd.to_numeric(df[tgt], errors="coerce").notna().mean() > 0.95,
             "연속형 종속변수는 숫자여야 합니다.")
        classes = None
    else:
        vc = df[tgt].dropna().astype(str).value_counts()
        need(len(vc) >= 2, "범주형 종속변수는 2개 이상의 클래스가 필요합니다.")
        need(len(vc) <= 50, f"클래스가 {len(vc)}개로 너무 많습니다. 연속형으로 분석하거나 범주를 병합하세요.")
        classes = sorted(vc.index.tolist(), key=lambda x: (pd.to_numeric(x, errors="coerce") if pd.notna(pd.to_numeric(x, errors="coerce")) else float("inf"), x))
        ttype = "binary" if len(classes) == 2 else "multiclass"
        if ttype == "binary":
            pos = str(p.get("positive") or (classes[1]))
            classes = [c for c in classes if c != pos] + [pos]
    roles = ws.st["roles"]
    old = ws.st.get("target")
    if old and old in roles and old != tgt:
        roles[old] = "feature"
    roles[tgt] = "target"
    ws.update(target=tgt, target_type=ttype, classes=classes, roles=roles, split=None, models=[], best_model=None)
    ws.mark_change(4, f"종속변수: {tgt} ({T.TYPE_LABEL[ttype]})")
    blocks = T.analyze(df, tgt, ttype)
    if ttype == "binary":
        blocks.insert(1, B.note(f"양성(관심) 클래스: `{classes[1]}` — 재현율·정밀도·AUC·Counterfactual 목표의 기준이 됩니다.", "info"))
    return "s4", blocks


# ====================================================== 5단계
@action("s5_overview")
def s5_overview(ws: WS, p: dict, job):
    df = ws.view()
    out = MS.overview(df, ws.st.get("target"), ws.st.get("target_type"))
    rec = MS.recommend(df)
    if rec:
        out.append(B.table(pd.DataFrame(rec), "변수별 권장 처리"))
    learned = [o for o in ws.st["learned_ops"] if o["op"] in ("impute", "indicator")]
    if learned:
        out.insert(0, B.note("적용된 결측 처리(위 진단은 처리 후 상태):\n" + "\n".join("• " + R.describe(o) for o in learned), "info"))
    return "s5", out


@action("s5_apply", mutating=True)
def s5_apply(ws: WS, p: dict, job):
    cols = p.get("columns") or []
    need(cols, "처리할 변수를 선택하세요.")
    method = p["method"]
    before = ws.view()
    b_shape = before.shape
    b_missing = int(before[cols].isna().sum().sum())
    if method == "drop":
        rec = ws.st["recipe"] + [{"op": "drop_missing", "columns": cols}]
        ws.update(recipe=rec)
        ws.rebuild(5, f"결측 행 삭제: {', '.join(cols)}")
    else:
        if method == "indicator":
            op = {"op": "indicator", "columns": cols}
        else:
            op = {"op": "impute", "columns": cols, "method": method}
            if method == "constant":
                v = p.get("value", 0)
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    pass
                op["value"] = v
            if method == "knn":
                op["k"] = int(p.get("k", 5))
        ops = ws.st["learned_ops"] + [op]
        ws.update(learned_ops=ops)
        ws.mark_change(5, R.describe(op))
    after = ws.view()
    out = [B.note("결측치 처리 완료", "success"),
           B.metrics([("처리 전 크기", f"{b_shape[0]:,}×{b_shape[1]}"), ("처리 후 크기", f"{after.shape[0]:,}×{after.shape[1]}"),
                      ("대상 열 결측(전)", b_missing), ("대상 열 결측(후)", int(after[[c for c in cols if c in after]].isna().sum().sum()))])]
    if method not in ("drop", "indicator"):
        out.append(B.note("이 대체는 '학습형 처리'로 기록되었습니다. 화면의 탐색용 데이터에는 전체 기준으로 미리 적용하고, "
                          "모델링에서는 8단계 분할 후 학습 데이터의 값으로 다시 계산해 평가 데이터에 적용합니다(누수 방지).", "info"))
    for c in cols[:4]:
        if pd.api.types.is_numeric_dtype(before[c]) and c in after:
            out += OL.compare_after(before[c], after[c], c)
    _, ov = s5_overview(ws, p, job)
    out += [B.heading("처리 후 결측 진단")] + ov
    return "s5", out


@action("s5_undo", mutating=True)
def s5_undo(ws: WS, p: dict, job):
    ops = ws.st["learned_ops"]
    i = int(p["index"])
    removed = ops.pop(i)
    ws.update(learned_ops=ops)
    ws.mark_change(5, f"학습형 처리 삭제: {R.describe(removed)}")
    return s5_overview(ws, p, job)


# ====================================================== 6단계
@action("s6_detect")
def s6_detect(ws: WS, p: dict, job):
    df = ws.view()
    col = p["column"]
    blocks, _ = OL.detect(df, col, p.get("method", "iqr"), p.get("factor"), p.get("lower"), p.get("upper"),
                          p.get("lower_pct", 1), p.get("upper_pct", 99))
    return "s6", blocks


@action("s6_apply", mutating=True)
def s6_apply(ws: WS, p: dict, job):
    cols = p.get("columns") or ([p["column"]] if p.get("column") else [])
    need(cols, "처리할 변수를 선택하세요.")
    method, action_ = p.get("method", "iqr"), p.get("action", "winsor")
    before = ws.view()
    out = []
    if action_ == "none":
        return "s6", [B.note("처리하지 않음을 선택했습니다.", "info")]
    if action_ == "drop":
        rec = ws.st["recipe"]
        for c in cols:
            lo, hi = R.bounds(before[c], method, p.get("factor"), p.get("lower"), p.get("upper"), p.get("lower_pct", 1), p.get("upper_pct", 99))
            rec = rec + [{"op": "drop_outliers", "column": c, "lower": lo, "upper": hi}]
        ws.update(recipe=rec)
        ws.rebuild(6, f"이상치 행 제거: {', '.join(cols)}")
    else:
        op = {"op": "clip", "columns": cols, "method": method, "mode": action_}
        for k in ("factor", "lower", "upper", "lower_pct", "upper_pct"):
            if p.get(k) not in (None, ""):
                op[k] = float(p[k])
        ws.update(learned_ops=ws.st["learned_ops"] + [op])
        ws.mark_change(6, R.describe(op))
    after = ws.view()
    out.append(B.note("이상치 처리 완료", "success"))
    out.append(B.metrics([("처리 전 크기", f"{before.shape[0]:,}×{before.shape[1]}"), ("처리 후 크기", f"{after.shape[0]:,}×{after.shape[1]}")]))
    for c in cols[:6]:
        out += OL.compare_after(before[c], after[c], c)
    if action_ != "drop":
        out.append(B.note("경계값은 모델링 시 학습 데이터에서 다시 계산합니다(누수 방지).", "info"))
    return "s6", out


@action("s6_multi")
def s6_multi(ws: WS, p: dict, job):
    return "s6_multi", OL.multivariate(ws.view(), p["columns"], p.get("method", "iforest"), float(p.get("contamination", 0.02)))


# ====================================================== 7단계
@action("s7")
def s7(ws: WS, p: dict, job):
    df = ws.view()
    kind = p["kind"]
    tgt, ttype = ws.st.get("target"), ws.st.get("target_type")
    n = int(p.get("max_rows", 2000))
    if kind == "scatter":
        blocks = eda.scatter(df, p["x"], p["y"], p.get("color") or None, p.get("trend", "ols"), n, p.get("hover") or None)
    elif kind == "box":
        blocks = eda.box(df, p["cat"], p["num"], int(p.get("top_n", 20)))
    elif kind == "corr":
        cols = p.get("columns") or eda.num_cols(df)[:12]
        need(len(cols) >= 2, "변수를 2개 이상 선택하세요.")
        blocks = eda.correlation(df, cols, p.get("method", "pearson"), tgt if tgt in cols else None)
    elif kind == "partial":
        blocks = eda.partial_corr(df, p["x"], p["y"], p.get("controls") or [])
    elif kind == "anova":
        need(tgt and ttype == "regression", "ANOVA는 종속변수가 연속형일 때 계산합니다(4단계).")
        blocks = eda.anova_all(df, tgt, p.get("columns") or eda.cat_cols(df, exclude=[tgt])[:10], int(p.get("max_groups", 30)))
    elif kind == "classtarget":
        need(tgt and ttype != "regression", "범주형 종속변수일 때 사용합니다.")
        feats = ws.features()
        blocks = eda.class_target(df, tgt, [c for c in feats if c in eda.num_cols(df)], [c for c in feats if c in eda.cat_cols(df)])
    elif kind == "crosstab":
        blocks = eda.crosstab(df, p["a"], p["b"])
    elif kind == "vif":
        cols = p.get("columns") or [c for c in ws.features() if c in eda.num_cols(df)]
        blocks = eda.vif(df, cols)
    elif kind == "mi":
        need(tgt, "4단계에서 종속변수를 먼저 선택하세요.")
        blocks = eda.mutual_info(df, tgt, ttype, p.get("columns") or ws.features())
    elif kind == "matrix":
        cols = p.get("columns") or eda.num_cols(df)[:5]
        blocks = eda.scatter_matrix(df, cols[:8], p.get("color") or None)
    elif kind == "trend":
        blocks = eda.trend(df, p["time"], p["values"], p.get("freq", "D"), p.get("agg", "mean"))
    elif kind == "summary":
        need(tgt, "4단계에서 종속변수를 먼저 선택하세요.")
        blocks = eda.related_summary(df, tgt, ttype, ws.features(), int(p.get("top", 10)))
    else:
        raise ValueError(kind)
    return f"s7_{kind}", blocks


# ====================================================== 8단계
@action("s8_split", mutating=True)
def s8_split(ws: WS, p: dict, job):
    tgt, ttype = ws.st.get("target"), ws.st.get("target_type")
    need(tgt, "4단계에서 종속변수를 먼저 선택하세요.")
    base = ws.modeling_base()
    feats = [f for f in (p.get("features") or ws.features()) if f in base.columns and f != tgt]
    need(feats, "독립변수를 하나 이상 선택하세요.")
    roles = ws.st["roles"]
    for c in base.columns:
        if c == tgt:
            continue
        if c in feats:
            roles[c] = "feature"
        elif roles.get(c) == "feature":
            roles[c] = "exclude"
    method = p.get("method", "stratified" if ttype != "regression" else "random")
    tr, te, info = MD.split_data(base, tgt, feats, ttype, method, float(p.get("test_size", 0.3)), int(p.get("seed", 42)),
                                 p.get("group_col"), p.get("time_col"), bool(p.get("dropna", False)))
    ws.p.save_obj("split_idx", {"train": tr, "test": te})
    sp = {"features": feats, "method": method, "test_size": float(p.get("test_size", 0.3)), "seed": int(p.get("seed", 42)),
          "group_col": p.get("group_col"), "time_col": p.get("time_col"), "n_train": len(tr), "n_test": len(te), **info}
    ws.update(split=sp, roles=roles, models=[], best_model=None, preprocess=None)
    ws.mark_change(8, f"데이터 분할: {method}, 학습 {len(tr):,} / 평가 {len(te):,}")
    return "s8", MD.split_blocks(base, tr, te, tgt, ttype, feats, info, p.get("group_col"))


# ====================================================== 9단계
@action("s9_preprocess", mutating=True)
def s9_preprocess(ws: WS, p: dict, job):
    Xtr, Xte, ytr, yte, _ = ws.xy()
    cfg = {"scaler": p.get("scaler", "standard"), "top_n": int(p.get("top_n", 5)), "encoding": p.get("encoding", "onehot"),
           "as_categorical": p.get("as_categorical") or [], "num_impute": p.get("num_impute", "median"), "drop_binary": bool(p.get("drop_binary"))}
    learned = ws.st["learned_ops"]
    pre = MD.build_preprocessor(cfg, learned)
    if cfg["encoding"] == "target":
        pre.fit(Xtr, ytr)
        blocks = [B.note("타깃 인코딩은 교차 적합되므로 미리보기 대신 모델 학습 시 적용됩니다.", "info")]
    else:
        blocks = MD.preprocess_blocks(pre, Xtr, Xte, cfg, learned)
    ws.update(preprocess=cfg, models=[], best_model=None)
    ws.mark_change(9, "모델링 전처리 설정")
    return "s9", blocks


# ====================================================== 10단계
def get_pre(ws: WS):
    cfg = ws.st.get("preprocess") or {"scaler": "standard", "top_n": 5, "encoding": "onehot"}
    return MD.build_preprocessor(cfg, ws.st["learned_ops"])


@action("s10_train", mutating=True)
def s10_train(ws: WS, p: dict, job):
    st = ws.st
    need(st.get("split"), "8단계에서 데이터를 분할하세요.")
    if not st.get("preprocess"):
        ws.update(preprocess={"scaler": "standard", "top_n": 5, "encoding": "onehot"})
    Xtr, Xte, ytr, yte, groups = ws.xy()
    ttype, classes = st["target_type"], st.get("classes") or []
    keys = p.get("models") or [m["key"] for m in MD.model_list(ttype) if m["tier"] == "기준"]
    if "baseline" not in keys:
        keys = ["baseline"] + keys
    if ttype == "regression":
        pass
    out = MD.train_models(get_pre(ws), keys, Xtr, ytr, Xte, yte, ttype, classes, st["split"]["method"], groups,
                          int(p.get("folds", 5)), int(st["split"].get("seed", 42)), bool(p.get("tune")), int(p.get("n_iter", 15)),
                          p.get("select_metric"), bool(p.get("class_weight")),
                          progress=job.progress, cancelled=job.is_cancelled)
    need(out["results"], "학습된 모델이 없습니다.")
    summary = []
    for k, r in out["results"].items():
        ws.p.save_obj(f"pipe_{k}", r["pipe"])
        summary.append({"key": k, "name": r["name"], "test": r["test"], "train": r["train"],
                        "cv": {m: {"mean": v["mean"], "std": v["std"]} for m, v in r["cv"].items()}, "best_params": r["best_params"]})
    ws.p.save_obj("train_out", {k: {kk: vv for kk, vv in r.items() if kk != "pipe"} for k, r in out["results"].items()})
    ws.update(models=summary, best_model=out["best"], select_metric=out["select_metric"], train_cfg=p)
    conformal(ws, out["best"])
    ws.mark_change(10, f"모델 학습: {len(summary)}개, 최적 {out['results'][out['best']]['name']}")
    return "s10", MD.training_blocks(out, ttype, classes)


def conformal(ws: WS, key: str):
    """회귀 예측구간용: 학습 데이터 교차검증 잔차의 분위수(분할 컨포멀 근사)."""
    if ws.st["target_type"] != "regression":
        return
    from sklearn.base import clone
    from sklearn.model_selection import cross_val_predict
    Xtr, Xte, ytr, yte, groups = ws.xy()
    pipe = ws.p.load_obj(f"pipe_{key}")
    try:
        cv = MD.cv_splitter(ws.st["split"]["method"] if ws.st["split"]["method"] != "time" else "random", "regression", 5, 0)
        oof = cross_val_predict(clone(pipe), Xtr, ytr, cv=cv, groups=groups if ws.st["split"]["method"] == "group" else None, n_jobs=-1)
        res = np.abs(np.asarray(ytr) - oof)
        ws.update(conformal={"90": float(np.quantile(res, 0.9)), "95": float(np.quantile(res, 0.95)), "model": key})
    except Exception:
        ws.update(conformal=None)


@action("s10_select", mutating=True)
def s10_select(ws: WS, p: dict, job):
    key = p["key"]
    need(ws.p.has_obj(f"pipe_{key}"), "학습된 모델이 아닙니다.")
    ws.update(best_model=key)
    conformal(ws, key)
    ws.mark_change(10, f"최적 모델 수동 지정: {key}")
    name = next((m["name"] for m in ws.st["models"] if m["key"] == key), key)
    return "s10_select", [B.note(f"이후 단계(11~16)에서 사용할 모델을 **{name}**(으)로 지정했습니다.", "success")]


# ====================================================== 11~12단계
def train_out(ws: WS):
    need(ws.st.get("best_model"), "10단계에서 모델을 학습하세요.")
    res = ws.p.load_obj("train_out")
    return res, ws.st["best_model"]


@action("s11_compare")
def s11_compare(ws: WS, p: dict, job):
    res, best = train_out(ws)
    Xtr, Xte, ytr, yte, groups = ws.xy()
    ttype, classes = ws.st["target_type"], ws.st.get("classes") or []
    blocks = [B.note(f"최적 모델: **{res[best]['name']}**", "success")]
    rows = pd.DataFrame([{"모델명": r["name"], **r["test"]} for r in res.values()])
    blocks.append(B.table(rows, "평가 데이터 성능 비교"))
    blocks += EV.model_comparison(res, ttype, classes)
    if ttype == "regression":
        blocks += EV.compare_regression(res, best, Xte, yte)
    else:
        blocks += EV.compare_classification(res, best, yte, classes)
    if p.get("learning_curve"):
        job.progress(0.8, "학습곡선 계산 중")
        pipe = ws.p.load_obj(f"pipe_{best}")
        blocks += EV.learning_curve_blocks(pipe, Xtr, ytr, ttype, MD.cv_splitter(ws.st["split"]["method"], ttype, 5, 0))
    return "s11_compare", blocks


@action("s11_explain")
def s11_explain(ws: WS, p: dict, job):
    res, best = train_out(ws)
    key = p.get("model") or best
    pipe = ws.p.load_obj(f"pipe_{key}")
    Xtr, Xte, ytr, yte, _ = ws.xy()
    ttype, classes = ws.st["target_type"], ws.st.get("classes") or []
    kinds = p.get("kinds") or ["perm", "model", "shap", "pdp"]
    tabs = []
    t0 = time.time()
    if "perm" in kinds:
        job.progress(0.1, "순열 중요도 계산 중")
        b, t = EV.permutation_blocks(pipe, Xte, yte, ttype)
        ws.update(importance={r["변수"]: float(r["중요도(성능 감소)"]) for _, r in t.iterrows()})
        tabs.append(("순열 중요도", b))
    if "model" in kinds:
        b = EV.model_importance_blocks(pipe)
        if b:
            tabs.append(("모델 내장 중요도", b))
    if "shap" in kinds:
        job.progress(0.4, "SHAP 계산 중")
        b, _ = EV.shap_blocks(pipe, Xtr, Xte, ttype, classes)
        tabs.append(("SHAP", b))
    if "pdp" in kinds:
        job.progress(0.75, "부분의존도 계산 중")
        feats = p.get("pdp_features")
        if not feats:
            imp = ws.st.get("importance") or {}
            feats = sorted(imp, key=imp.get, reverse=True)[:4] if imp else list(Xtr.columns[:4])
        tabs.append(("부분의존도(PDP·ICE)", EV.pdp_blocks(pipe, Xtr, feats, ttype, classes)))
    return "s11_explain", [B.note(f"설명 대상 모델: **{res[key]['name']}**", "info"), B.tabs(tabs)]


# ====================================================== 13단계
def model_ctx(ws: WS, key: str | None = None) -> dict:
    res, best = train_out(ws)
    key = key or best
    st = ws.st
    feats = st["split"]["features"]
    base_cols = ws.base_columns(feats)
    base, tr, te, sp = ws.split_data()
    work = ws.working()
    tr_valid = [i for i in tr if i in work.index]
    base_train = work.loc[tr_valid, base_cols]
    spec = PR.input_spec(base_train, base_cols)
    Xtr, Xte, ytr, yte, _ = ws.xy()
    others = {}
    for k, r in res.items():
        if k not in (key, "baseline") and ws.p.has_obj(f"pipe_{k}"):
            others[k] = (r["name"], ws.p.load_obj(f"pipe_{k}"))
    return {"pipe": ws.p.load_obj(f"pipe_{key}"), "model_name": res[key]["name"], "ttype": st["target_type"],
            "classes": st.get("classes") or [], "features": feats, "recipe": st["recipe"], "spec": spec, "base_cols": base_cols,
            "base_train": base_train, "y_base_train": work.loc[tr_valid, st["target"]], "Xtr": Xtr, "ytr": ytr,
            "conformal": st.get("conformal") if (st.get("conformal") or {}).get("model") == key else None,
            "others": others, "target": st["target"], "test_metrics": res[key]["test"],
            "causal_effects": st.get("causal_effects")}


@action("s13_predict")
def s13_predict(ws: WS, p: dict, job):
    ctx = model_ctx(ws, p.get("model"))
    return "s13", PR.predict_one(ctx, p.get("row") or {})


@action("s13_whatif")
def s13_whatif(ws: WS, p: dict, job):
    ctx = model_ctx(ws, p.get("model"))
    return "s13_whatif", PR.whatif(ctx, p.get("row") or {}, p["var"])


@action("s13_sensitivity")
def s13_sensitivity(ws: WS, p: dict, job):
    ctx = model_ctx(ws, p.get("model"))
    return "s13_sens", PR.sensitivity_all(ctx, p.get("row") or {}) + [
        B.interpret("각 수치 입력을 학습 데이터의 하위 1%와 상위 99% 값으로 바꿨을 때 예측이 얼마나 움직이는지 비교했습니다. "
                    "막대가 긴 변수일수록 이 조건에서 예측에 영향이 큰 변수입니다.")]


# ====================================================== 14단계
def causal_frame(ws: WS) -> pd.DataFrame:
    return ws.view()


@action("s14_causal")
def s14_causal(ws: WS, p: dict, job):
    st = ws.st
    need(st.get("target"), "4단계에서 종속변수를 먼저 선택하세요.")
    tgt, ttype, classes = st["target"], st["target_type"], st.get("classes") or []
    treat = p["treatment"]
    controls = [c for c in (p.get("controls") or []) if c not in (treat, tgt)]
    df = causal_frame(ws)
    d, info = C.prepare(df, treat, tgt, controls, ttype, classes, p.get("outcome_class"), bool(p.get("standardize")), p.get("binarize") or None)
    method = p.get("method", "backdoor.linear_regression")
    job.progress(0.1, "DoWhy Model·Identify·Estimate 실행 중")
    refuters = p.get("refuters") or ["random_common_cause", "placebo_treatment_refuter", "data_subset_refuter"]
    extra = {}
    try:
        res = C.run_dowhy(d, info, treat, tgt, method, p.get("edges") or None, refuters)
    except ImportError:
        raise ValueError("DoWhy가 설치되어 있지 않습니다. `pip install dowhy`를 실행하세요.")
    except Exception as e:
        fit, eff, lo, hi, pv = C.ols_effect(d, treat, tgt, info["controls"])
        return "s14", [B.note(f"DoWhy 실행 중 오류가 발생해 통제변수 OLS로 대체 추정했습니다(반박 검정 없음): {e}", "warn"),
                       B.metrics([("OLS 대체 추정 효과", eff), ("95% 신뢰구간", f"{lo:.4g} ~ {hi:.4g}"), ("p-value", pv)]),
                       B.code(fit.summary().as_text())]
    job.progress(0.7, "보조 추정(OLS·DML) 중")
    try:
        extra["ols"] = C.ols_effect(d, treat, tgt, info["controls"])
    except Exception:
        pass
    if p.get("dml", True) and info["controls"]:
        try:
            extra["dml"] = C.dml_effect(d, treat, tgt, info["controls"])
        except Exception:
            pass
    try:
        import statsmodels.api as sm
        extra["naive"] = float(sm.OLS(d[tgt], sm.add_constant(d[[treat]])).fit().params[treat])
    except Exception:
        pass
    blocks = C.single_blocks(res, d, info, treat, tgt, method, extra)
    msgs = []
    if info.get("n_imputed"):
        msgs.append(f"처치·통제변수의 남은 결측 {info['n_imputed']:,}셀은 중앙값으로 채웠습니다(5단계에서 처리 방법을 직접 정할 수 있습니다).")
    if info.get("n_dropped"):
        msgs.append(f"결과변수 결측 등으로 {info['n_dropped']:,}행을 제외했습니다.")
    if msgs:
        blocks.insert(1, B.note(" ".join(msgs), "info"))
    return "s14", blocks


@action("s14_batch")
def s14_batch(ws: WS, p: dict, job):
    st = ws.st
    need(st.get("target"), "4단계에서 종속변수를 먼저 선택하세요.")
    cands = p.get("candidates") or ws.features()
    df = causal_frame(ws)
    std = bool(p.get("standardize", True))
    blocks = C.batch_effects(df, cands, st["target"], st["target_type"], st.get("classes") or [], st.get("importance"),
                             p.get("outcome_class"), std, job.progress)
    # 원 단위 효과를 Counterfactual 검증용으로 저장
    try:
        tab = next(b for b in blocks if b["type"] == "table")
        ci = tab["columns"].index("변수")
        ei = tab["columns"].index("원 단위 효과")
        ws.update(causal_effects={r[ci]: r[ei] for r in tab["rows"] if r[ei] is not None})
    except Exception:
        pass
    if not st.get("importance"):
        blocks.append(B.note("11~12단계에서 '순열 중요도'를 실행하면 예측 중요도와 인과효과를 나란히 비교할 수 있습니다.", "info"))
    return "s14_batch", blocks


@action("s14_hetero")
def s14_hetero(ws: WS, p: dict, job):
    st = ws.st
    return "s14_hetero", C.heterogeneity(causal_frame(ws), p["treatment"], st["target"], p.get("controls") or [], p["segment"],
                                         st["target_type"], st.get("classes") or [], p.get("outcome_class"), bool(p.get("standardize")))


# ====================================================== 15단계
def rule_of(p, ttype):
    if ttype == "regression":
        return {"direction": p.get("direction", "이하"), "target_value": float(p["target_value"])}
    return {"target_class": str(p["target_class"]), "min_prob": float(p.get("min_prob", 0.5))}


@action("s15_cf")
def s15_cf(ws: WS, p: dict, job):
    ctx = model_ctx(ws, p.get("model"))
    blocks, _ = CF.search(ctx, p.get("row") or {}, p.get("mutable") or [], rule_of(p, ctx["ttype"]), p.get("method", "random"),
                          int(p.get("n_samples", 3000)), int(p.get("top_k", 5)), int(p.get("seed", 42)), int(p.get("max_change", 4)),
                          p.get("constraints") or {}, True, job.progress, job.is_cancelled)
    return "s15", blocks


@action("s15_batch")
def s15_batch(ws: WS, p: dict, job):
    ctx = model_ctx(ws, p.get("model"))
    base, tr, te, sp = ws.split_data()
    work = ws.working()
    data = work.loc[[i for i in te if i in work.index], ctx["base_cols"]] if p.get("scope", "test") == "test" else work[ctx["base_cols"]]
    return "s15_batch", CF.batch(ctx, data, p.get("mutable") or [], rule_of(p, ctx["ttype"]), int(p.get("n_rows", 100)),
                                 int(p.get("n_samples", 400)), int(p.get("max_change", 3)), p.get("constraints") or {}, 0, job.progress, job.is_cancelled)


@action("s15_optimize")
def s15_optimize(ws: WS, p: dict, job):
    ctx = model_ctx(ws, p.get("model"))
    work = ws.working()
    return "s15_opt", CF.optimize(ctx, work[ctx["base_cols"]], p.get("mutable") or [], rule_of(p, ctx["ttype"]), int(p.get("n_samples", 300)),
                                  p.get("constraints") or {}, 0, job.progress, job.is_cancelled)


# ====================================================== 16단계
@action("s16_predict")
def s16_predict(ws: WS, p: dict, job):
    ctx = model_ctx(ws, p.get("model"))
    nd = next((n for n in ws.st.get("newdata", []) if n["id"] == p["file"]), None)
    need(nd, "신규 데이터 파일을 업로드하세요.")
    df = ws.p.load_df(f"new_{nd['id']}")
    blocks, result = ND.run(ctx, df, nd)
    ws.p.save_df(f"pred_{nd['id']}", result)
    blocks.insert(0, B.note(f"`{nd['name']}` 예측 완료 · 모델 {ctx['model_name']}", "success"))
    blocks.append({"type": "download", "label": "전체 예측 결과 CSV 내려받기", "href": f"download/pred_{nd['id']}"})
    return "s16", blocks


# ====================================================== 실행기

def run_action(pid: str, name: str, params: dict, job) -> tuple[str, list]:
    meta = REGISTRY.get(name)
    if not meta:
        raise ValueError(f"알 수 없는 작업: {name}")
    ws = WS(pid)
    if meta["mutating"]:
        with lock(pid):
            key, blocks = meta["fn"](ws, params, job)
    else:
        key, blocks = meta["fn"](ws, params, job)
    ws.p.save_result(key, blocks, params)
    return key, blocks
