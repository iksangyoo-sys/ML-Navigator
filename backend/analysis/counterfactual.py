"""15단계: Counterfactual XAI — 목표 예측을 만족하는 최소 변경 조건 탐색."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import blocks as B
from .charts import PALETTE, style
from .predict import coerce_row, make_X


class Space:
    """변경 가능한 변수의 탐색 공간(범위·방향·비용)."""

    def __init__(self, spec: list[dict], mutable: list[str], constraints: dict | None = None):
        self.spec = {s["name"]: s for s in spec}
        self.mutable = [m for m in mutable if m in self.spec]
        self.cons = constraints or {}

    def bounds(self, c, base_val):
        s = self.spec[c]
        k = self.cons.get(c, {})
        lo = float(k["min"]) if k.get("min") not in (None, "") else s["q01"]
        hi = float(k["max"]) if k.get("max") not in (None, "") else s["q99"]
        if lo == hi:
            lo, hi = s["min"], s["max"]
        d = k.get("direction", "both")
        if d == "increase":
            lo = max(lo, base_val)
        elif d == "decrease":
            hi = min(hi, base_val)
        return lo, max(hi, lo)

    def cost(self, c):
        try:
            return float(self.cons.get(c, {}).get("cost", 1) or 1)
        except (TypeError, ValueError):
            return 1.0

    def options(self, c):
        return self.spec[c].get("raw_options", [])[:20]


def predict_fn(pipe, ttype, classes):
    def f(X):
        if ttype == "regression":
            return pipe.predict(X), None
        pr = pipe.predict_proba(X)
        return pr.argmax(1), pr
    return f


def success_mask(pred, proba, ttype, rule, classes):
    if ttype == "regression":
        tv = float(rule["target_value"])
        return pred <= tv if rule["direction"] == "이하" else pred >= tv
    ci = classes.index(rule["target_class"]) if rule["target_class"] in classes else [str(c) for c in classes].index(str(rule["target_class"]))
    mp = float(rule.get("min_prob", 0.5))
    return (pred == ci) & (proba[:, ci] >= mp)


def target_value(pred, proba, ttype, rule, classes):
    if ttype == "regression":
        return pred
    ci = [str(c) for c in classes].index(str(rule["target_class"]))
    return proba[:, ci]


def distance(cands: pd.DataFrame, base: dict, space: Space) -> tuple[np.ndarray, np.ndarray]:
    dist = np.zeros(len(cands))
    nchg = np.zeros(len(cands), dtype=int)
    for c in space.mutable:
        s = space.spec[c]
        if s["type"] == "number":
            rng = max(s["q99"] - s["q01"], 1e-9)
            diff = np.abs(cands[c].astype(float).values - float(base[c]))
            changed = diff > 1e-9 * max(1, abs(float(base[c])))
            dist += np.where(changed, diff / rng, 0) * space.cost(c)
            nchg += changed
        else:
            changed = cands[c].astype(str).values != str(base[c])
            dist += changed * 1.0 * space.cost(c)
            nchg += changed
    return dist, nchg


def random_candidates(base: dict, space: Space, n: int, rng, max_change: int) -> pd.DataFrame:
    rows = pd.DataFrame([base] * n)
    m = len(space.mutable)
    k_max = max(1, min(m, max_change))
    for i in range(n):
        k = int(rng.integers(1, k_max + 1))
        chosen = rng.choice(space.mutable, size=k, replace=False)
        for c in chosen:
            s = space.spec[c]
            if s["type"] == "number":
                lo, hi = space.bounds(c, float(base[c]))
                if rng.random() < 0.7:
                    v = float(base[c]) + rng.normal(0, (s["std"] or (hi - lo) / 4) * 0.5)
                    v = float(np.clip(v, lo, hi))
                else:
                    v = float(rng.uniform(lo, hi))
                if s.get("integer"):
                    v = round(v)
                rows.at[i, c] = v
            else:
                opts = space.options(c)
                if opts:
                    rows.at[i, c] = opts[int(rng.integers(0, len(opts)))]
    return rows


def genetic(base: dict, space: Space, f, rule, ttype, classes, recipe, features, rng, pop=200, gens=25, max_change=4, progress=None, cancelled=None):
    """목표 달성 + 거리 최소화를 함께 최적화하는 간단한 유전 알고리즘."""
    popu = random_candidates(base, space, pop, rng, max_change)
    archive = []

    def fitness(P):
        X = make_X(P, recipe, features)
        pred, proba = f(X)
        ok = success_mask(pred, proba, ttype, rule, classes)
        tv = target_value(pred, proba, ttype, rule, classes)
        dist, nchg = distance(P, base, space)
        if ttype == "regression":
            gap = np.maximum(0, (tv - float(rule["target_value"])) if rule["direction"] == "이하" else (float(rule["target_value"]) - tv))
            scale = abs(float(rule["target_value"])) or 1
            gap = gap / scale
        else:
            gap = np.maximum(0, float(rule.get("min_prob", 0.5)) - tv)
        penalty = np.where(nchg > max_change, 10, 0)
        return dist + 20 * gap + penalty, ok, pred, proba, dist, nchg
    for g in range(gens):
        if cancelled and cancelled():
            break
        if progress:
            progress(g / gens, f"유전 알고리즘 {g + 1}/{gens}세대")
        fit, ok, pred, proba, dist, nchg = fitness(popu)
        if ok.any():
            a = popu[ok].copy()
            a["__pred"] = target_value(pred, proba, ttype, rule, classes)[ok]
            a["__dist"] = dist[ok]
            a["__nchg"] = nchg[ok]
            archive.append(a)
        order = np.argsort(fit)
        elite = popu.iloc[order[: pop // 4]].reset_index(drop=True)
        children = []
        for _ in range(pop - len(elite)):
            p1, p2 = elite.iloc[int(rng.integers(0, len(elite)))], elite.iloc[int(rng.integers(0, len(elite)))]
            child = p1.copy()
            for c in space.mutable:
                if rng.random() < 0.5:
                    child[c] = p2[c]
                if rng.random() < 0.15:  # 돌연변이
                    s = space.spec[c]
                    if s["type"] == "number":
                        lo, hi = space.bounds(c, float(base[c]))
                        v = float(np.clip(float(child[c]) + rng.normal(0, (hi - lo) * 0.1 + 1e-12), lo, hi))
                        child[c] = round(v) if s.get("integer") else v
                    else:
                        opts = space.options(c)
                        if opts:
                            child[c] = opts[int(rng.integers(0, len(opts)))]
                if rng.random() < 0.1:  # 원래 값으로 되돌려 변경 수를 줄이는 방향
                    child[c] = base[c]
            children.append(child)
        popu = pd.concat([elite, pd.DataFrame(children)], ignore_index=True)
    return pd.concat(archive, ignore_index=True) if archive else pd.DataFrame()


def search(ctx: dict, row: dict, mutable: list[str], rule: dict, method="random", n_samples=3000, top_k=5, seed=42,
           max_change=4, constraints=None, diversity=True, progress=None, cancelled=None) -> tuple[list, pd.DataFrame]:
    pipe, ttype, classes, spec, recipe, features = ctx["pipe"], ctx["ttype"], ctx["classes"], ctx["spec"], ctx["recipe"], ctx["features"]
    base = coerce_row(row, spec)
    space = Space(spec, mutable, constraints)
    if not space.mutable:
        raise ValueError("변경 가능한 변수를 하나 이상 선택하세요.")
    rng = np.random.default_rng(seed)
    f = predict_fn(pipe, ttype, classes)
    bX = make_X(pd.DataFrame([base]), recipe, features)
    bp, bpr = f(bX)
    base_val = float(target_value(bp, bpr, ttype, rule, classes)[0])
    reach = None
    if method == "genetic":
        found = genetic(base, space, f, rule, ttype, classes, recipe, features, rng, pop=max(100, n_samples // 20), gens=25,
                        max_change=max_change, progress=progress, cancelled=cancelled)
    else:
        found = []
        reach = None
        batch = 1000
        done = 0
        while done < n_samples:
            if cancelled and cancelled():
                break
            if progress:
                progress(done / n_samples, f"후보 {done:,}/{n_samples:,} 탐색")
            m = min(batch, n_samples - done)
            P = random_candidates(base, space, m, rng, max_change)
            X = make_X(P, recipe, features)
            pred, proba = f(X)
            ok = success_mask(pred, proba, ttype, rule, classes)
            tvals = target_value(pred, proba, ttype, rule, classes)
            want_low = ttype == "regression" and rule.get("direction") == "이하"
            cand = float(np.min(tvals) if want_low else np.max(tvals))
            reach = cand if reach is None else (min(reach, cand) if want_low else max(reach, cand))
            if ok.any():
                a = P[ok].copy()
                dist, nchg = distance(a, base, space)
                a["__pred"] = target_value(pred, proba, ttype, rule, classes)[ok]
                a["__dist"] = dist
                a["__nchg"] = nchg
                found.append(a)
            done += m
        found = pd.concat(found, ignore_index=True) if found else pd.DataFrame()
    if found.empty:
        return [B.metrics([("현재 예측", base_val)]),
                B.note("목표를 만족하는 대안을 찾지 못했습니다."
                       + (f" 탐색한 후보 중 도달한 가장 좋은 예측은 {reach:.4g}입니다. 목표를 이 근처로 완화하거나," if reach is not None else " 목표를 완화하거나,")
                       + " 변경 가능한 변수·허용 범위·최대 변경 수를 늘려 보세요.", "warn")], pd.DataFrame()
    found = found.drop_duplicates(subset=space.mutable).sort_values(["__dist", "__nchg"])
    # 다양성: 같은 변수 조합만 반복되지 않도록
    picked = []
    if diversity:
        seen = set()
        for _, r in found.iterrows():
            sig = tuple(sorted(c for c in space.mutable if (abs(float(r[c]) - float(base[c])) > 1e-9 if space.spec[c]["type"] == "number" else str(r[c]) != str(base[c]))))
            if sig in seen and len(picked) < top_k * 3:
                continue
            seen.add(sig)
            picked.append(r)
            if len(picked) >= top_k:
                break
        best = pd.DataFrame(picked)
    else:
        best = found.head(top_k)
    if len(best) < top_k:
        best = pd.concat([best, found[~found.index.isin(best.index)].head(top_k - len(best))])
    summary = []
    for i, (_, r) in enumerate(best.iterrows()):
        changes = []
        for c in space.mutable:
            s = space.spec[c]
            if s["type"] == "number":
                if abs(float(r[c]) - float(base[c])) > 1e-9:
                    d = float(r[c]) - float(base[c])
                    changes.append(f"{c}: {float(base[c]):.4g} → {float(r[c]):.4g} ({'+' if d > 0 else ''}{d:.3g})")
            elif str(r[c]) != str(base[c]):
                changes.append(f"{c}: {base[c]} → {r[c]}")
        summary.append({"순위": i + 1, "거리점수": float(r["__dist"]), "예측값" if ttype == "regression" else "목표 클래스 확률": float(r["__pred"]),
                        "변경개수": int(r["__nchg"]), "변경 제안": " / ".join(changes) or "변경 없음"})
    st = pd.DataFrame(summary)
    # 근접 실측 사례: 추천안과 가장 가까운 실제 학습 데이터 행
    Xtr_base = ctx.get("base_train")
    near_rows = []
    if Xtr_base is not None and len(Xtr_base):
        num = [c for c in space.mutable if space.spec[c]["type"] == "number"]
        if num:
            M = Xtr_base[num].astype(float)
            sd = M.std().replace(0, 1)
            for i, (_, r) in enumerate(best.iterrows()):
                dd = (((M - r[num].astype(float)) / sd) ** 2).sum(axis=1) ** 0.5
                j = dd.idxmin()
                near_rows.append({"순위": i + 1, "근접 실측 거리(표준화)": float(dd.min()),
                                  "실측 타깃": ctx["y_base_train"].loc[j] if ctx.get("y_base_train") is not None else None,
                                  "판정": "실측 근거 있음" if dd.min() < 0.5 * np.sqrt(len(num)) else "실측 사례 드묾(외삽 주의)"})
    detail = best[[c for c in best.columns if not c.startswith("__")]].copy()
    detail.insert(0, "순위", range(1, len(detail) + 1))
    # 변경량 시각화
    fig = go.Figure()
    for c in space.mutable:
        s = space.spec[c]
        if s["type"] != "number":
            continue
        rngv = max(s["q99"] - s["q01"], 1e-9)
        fig.add_trace(go.Bar(name=c, x=[f"#{i + 1}" for i in range(len(best))], y=(best[c].astype(float) - float(base[c])) / rngv * 100))
    fig.update_layout(barmode="group")
    style(fig, "추천안별 변경량 (학습 범위 대비 %)", "추천안", "변경량(%)", height=360)
    out = [
        B.metrics([("현재 예측" if ttype == "regression" else "현재 목표 확률", base_val), ("목표 만족 후보", len(found)),
                   ("최소 거리점수", float(st["거리점수"].min())), ("최소 변경 개수", int(st["변경개수"].min()))]),
        B.table(st, "추천 변경안 요약", download="counterfactual_summary.csv"),
        B.chart(fig),
        B.table(detail, "추천안 상세 (변경 가능 변수 외는 현재값 유지)", download="counterfactual_candidates.csv"),
    ]
    if near_rows:
        out.append(B.table(pd.DataFrame(near_rows), "추천안 검증: 근접 실측 사례"))
    ce = ctx.get("causal_effects")
    if ce:
        rows = []
        for c in space.mutable:
            if c in ce and space.spec[c]["type"] == "number":
                d = float(best.iloc[0][c]) - float(base[c])
                if abs(d) < 1e-12:
                    continue
                e = ce[c]
                want = -1 if (ttype == "regression" and rule.get("direction") == "이하") else 1
                agree = np.sign(d * e) == want
                rows.append({"변수": c, "변경 방향": "증가" if d > 0 else "감소", "인과효과 부호": "+" if e > 0 else "-",
                             "인과 근거와 일치": "일치" if agree else "불일치(모델 상관에 의존 가능)"})
        if rows:
            out.append(B.table(pd.DataFrame(rows), "1순위 추천안과 14단계 인과효과 방향 비교"))
    b = st.iloc[0]
    out.append(B.interpret(
        f"가장 가까운 대안은 변경 {int(b['변경개수'])}개, 거리점수 {b['거리점수']:.4f}입니다: {b['변경 제안']}. "
        "거리점수는 각 변수의 변경량을 학습 데이터 1~99% 범위로 나눈 값의 합(범주형은 변경 시 1점)에 변경 비용 가중치를 곱한 것입니다. "
        "Counterfactual은 '모델이 목표를 예측하게 만드는 최소 변경'이므로, 실제 적용 전에 근접 실측 사례와 인과효과 방향이 맞는지 확인하고 "
        "소규모 확인 실험으로 검증하는 것을 권합니다."))
    return out, detail


def batch(ctx: dict, base_df: pd.DataFrame, mutable, rule, n_rows=100, n_samples=500, max_change=3, constraints=None, seed=0,
          progress=None, cancelled=None) -> list:
    pipe, ttype, classes, spec, recipe, features = ctx["pipe"], ctx["ttype"], ctx["classes"], ctx["spec"], ctx["recipe"], ctx["features"]
    f = predict_fn(pipe, ttype, classes)
    X = make_X(base_df, recipe, features)
    pred, proba = f(X)
    ok = success_mask(pred, proba, ttype, rule, classes)
    fail = base_df[~ok]
    if fail.empty:
        return [B.note("모든 행이 이미 목표를 만족합니다.", "success")]
    targets = fail.sample(min(n_rows, len(fail)), random_state=seed)
    space = Space(spec, mutable, constraints)
    rng = np.random.default_rng(seed)
    results = []
    counts = {c: 0 for c in space.mutable}
    deltas = {c: [] for c in space.mutable}
    for i, (idx, r) in enumerate(targets.iterrows()):
        if cancelled and cancelled():
            break
        if progress:
            progress(i / len(targets), f"행 {i + 1}/{len(targets)} 개선안 탐색")
        base = coerce_row(r.to_dict(), spec)
        P = random_candidates(base, space, n_samples, rng, max_change)
        Xp = make_X(P, recipe, features)
        pp, ppr = f(Xp)
        okp = success_mask(pp, ppr, ttype, rule, classes)
        if not okp.any():
            results.append({"행": idx, "개선안": "찾지 못함", "변경개수": None, "거리": None})
            continue
        dist, nchg = distance(P[okp], base, space)
        j = int(np.lexsort((nchg, dist))[0])
        bestc = P[okp].iloc[j]
        chg = []
        for c in space.mutable:
            if space.spec[c]["type"] == "number":
                d = float(bestc[c]) - float(base[c])
                if abs(d) > 1e-9:
                    counts[c] += 1
                    deltas[c].append(d)
                    chg.append(f"{c} {'+' if d > 0 else ''}{d:.3g}")
            elif str(bestc[c]) != str(base[c]):
                counts[c] += 1
                chg.append(f"{c}→{bestc[c]}")
        results.append({"행": idx, "개선안": ", ".join(chg), "변경개수": int(nchg[j]), "거리": float(dist[j])})
    rt = pd.DataFrame(results)
    solved = rt["변경개수"].notna().mean() if len(rt) else 0
    ct = pd.DataFrame([{"변수": c, "제안 횟수": n, "제안 비율(%)": n / max(len(rt), 1) * 100,
                        "평균 변경량": float(np.mean(deltas[c])) if deltas[c] else None} for c, n in counts.items()]).sort_values("제안 횟수", ascending=False)
    fig = go.Figure(go.Bar(x=ct["변수"], y=ct["제안 비율(%)"], marker_color=PALETTE[0]))
    style(fig, "개선안에 가장 자주 등장한 변수", None, "제안 비율(%)", height=340, legend=False)
    return [B.metrics([("목표 미달 행(전체)", len(fail)), ("분석한 행", len(targets)), ("개선안 찾은 비율(%)", round(solved * 100, 1))]),
            B.chart(fig), B.table(ct, "변수별 제안 빈도"), B.table(rt, "행별 개선안", download="batch_counterfactual.csv"),
            B.interpret("목표를 못 맞춘 행 각각에 대해 가장 가까운 개선안을 찾고, 어떤 변수가 가장 자주 조정 대상으로 제안되는지 집계했습니다. "
                        "제안 비율이 높은 변수가 공정 개선의 핵심 레버일 가능성이 큽니다. 평균 변경량의 부호로 일반적인 조정 방향을 확인하세요.")]


def optimize(ctx: dict, base_df: pd.DataFrame, mutable, rule, n_samples=300, constraints=None, seed=0, progress=None, cancelled=None) -> list:
    """모든 행에 같은 조정값을 적용했을 때 목표 달성률(또는 평균 예측)을 최대화하는 조건 탐색."""
    pipe, ttype, classes, spec, recipe, features = ctx["pipe"], ctx["ttype"], ctx["classes"], ctx["spec"], ctx["recipe"], ctx["features"]
    f = predict_fn(pipe, ttype, classes)
    space = Space(spec, mutable, constraints)
    rng = np.random.default_rng(seed)
    data = base_df.sample(min(len(base_df), 500), random_state=seed)
    X0 = make_X(data, recipe, features)
    p0, pr0 = f(X0)
    rate0 = float(success_mask(p0, pr0, ttype, rule, classes).mean())
    mean0 = float(np.mean(target_value(p0, pr0, ttype, rule, classes)))
    rows = []
    for i in range(n_samples):
        if cancelled and cancelled():
            break
        if progress and i % 20 == 0:
            progress(i / n_samples, f"조건 조합 {i}/{n_samples} 평가")
        setting = {}
        for c in space.mutable:
            s = space.spec[c]
            if s["type"] == "number":
                lo, hi = space.bounds(c, s["median"])
                v = float(rng.uniform(lo, hi))
                setting[c] = round(v) if s.get("integer") else v
            else:
                opts = space.options(c)
                setting[c] = opts[int(rng.integers(0, len(opts)))] if opts else None
        d = data.copy()
        for c, v in setting.items():
            d[c] = v
        Xd = make_X(d, recipe, features)
        p, pr = f(Xd)
        rate = float(success_mask(p, pr, ttype, rule, classes).mean())
        mval = float(np.mean(target_value(p, pr, ttype, rule, classes)))
        rows.append({**setting, "목표 달성률(%)": rate * 100, "평균 예측": mval})
    t = pd.DataFrame(rows)
    asc = ttype == "regression" and rule.get("direction") == "이하"
    t = t.sort_values(["목표 달성률(%)", "평균 예측"], ascending=[False, asc]).reset_index(drop=True)
    best = t.iloc[0]
    figs = []
    for c in [c for c in space.mutable if space.spec[c]["type"] == "number"][:6]:
        fig = go.Figure(go.Scattergl(x=t[c], y=t["목표 달성률(%)"], mode="markers", marker={"size": 5, "color": t["평균 예측"], "colorscale": "Viridis", "showscale": True}))
        style(fig, f"{c} 설정값별 목표 달성률", c, "달성률(%)", height=300, legend=False)
        figs.append(B.chart(fig))
    return [B.metrics([("현재 달성률(%)", rate0 * 100), ("최적 조건 달성률(%)", float(best["목표 달성률(%)"])),
                       ("현재 평균 예측", mean0), ("최적 조건 평균 예측", float(best["평균 예측"]))]),
            B.table(t.head(20), "목표 달성률 상위 조건 조합", download="optimal_settings.csv")] + figs + [
            B.interpret("조정 가능한 변수를 모든 행에 같은 값으로 설정한다고 가정하고, 나머지 조건(원료·환경 등)은 실제 데이터 그대로 둔 채 "
                        "목표 달성률이 가장 높은 설정을 찾았습니다. 공정 표준 조건을 정할 때 출발점으로 쓰되, 학습 데이터에서 드문 조합일 수 있으므로 "
                        "DOE 확인실험으로 검증하세요.")]
