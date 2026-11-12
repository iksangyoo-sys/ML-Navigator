"""프로젝트 작업공간: 데이터 단계별 접근, 캐시, 상태 요약."""
from __future__ import annotations

import re
import time
from collections import OrderedDict

import numpy as np
import pandas as pd

from .analysis import dataio, features as F, recipe as R
from .storage import Project, lock

_cache: "OrderedDict[tuple, pd.DataFrame]" = OrderedDict()
CACHE_MAX = 8

STEP_TITLES = {
    1: "데이터 불러오기", 2: "데이터 결합하기", 3: "파생변수 생성하기", 4: "종속변수 분포확인", 5: "결측치 처리",
    6: "이상치 처리", 7: "탐색적 분석", 8: "데이터 분할", 9: "모델링 전처리", 10: "모델 학습 및 평가",
    11: "모델 비교 및 시각화", 13: "직접 입력 예측", 14: "DoWhy 인과분석", 15: "Counterfactual XAI", 16: "신규 데이터 예측",
}


def step_of(key: str) -> int:
    m = re.match(r"s(\d+)", key)
    return int(m.group(1)) if m else 0


class WS:
    def __init__(self, pid: str):
        self.p = Project(pid)
        self.pid = pid

    # ------------------------------------------------ 상태
    @property
    def st(self) -> dict:
        return self.p.state

    def update(self, **kw) -> dict:
        return self.p.update_state(**kw)

    def mark_change(self, step: int, history: str | None = None) -> None:
        with lock(self.pid):
            st = self.p.state
            st.setdefault("changes", {})[str(step)] = time.time()
            if history:
                st["history"].append({"time": time.time(), "step": step, "text": history})
                st["history"] = st["history"][-200:]
            self.p.save_state(st)

    # ------------------------------------------------ 캐시
    def _key(self, name):
        return (self.pid, name, self.st.get("changes", {}).__repr__())

    def cached(self, name: str, loader):
        k = (self.pid, name, repr(sorted(self.st.get("changes", {}).items())), repr(self.st.get("learned_ops")))
        if k in _cache:
            _cache.move_to_end(k)
            return _cache[k]
        df = loader()
        _cache[k] = df
        while len(_cache) > CACHE_MAX:
            _cache.popitem(last=False)
        return df

    # ------------------------------------------------ 데이터
    def source_df(self, sid: str) -> pd.DataFrame:
        return self.cached(f"src_{sid}", lambda: self.p.load_df(f"src_{sid}"))

    def sources_named(self, ids: list[str] | None = None) -> list[tuple[str, pd.DataFrame]]:
        srcs = self.st["sources"]
        if ids:
            srcs = [s for sid in ids for s in srcs if s["id"] == sid]
        return [(s["name"], self.source_df(s["id"])) for s in srcs]

    def has_merged(self) -> bool:
        return self.p.has_df("merged")

    def merged(self) -> pd.DataFrame:
        if not self.p.has_df("merged"):
            srcs = self.st["sources"]
            if not srcs:
                raise ValueError("먼저 1단계에서 데이터를 불러오세요.")
            self.set_merged(self.source_df(srcs[0]["id"]), {"mode": "single", "sources": [srcs[0]["id"]]}, step=1)
        return self.cached("merged", lambda: self.p.load_df("merged"))

    def set_merged(self, df: pd.DataFrame, cfg: dict, step: int = 2) -> list[str]:
        self.p.save_df("merged", df)
        self.update(merge=cfg)
        return self.rebuild(step, f"결합 데이터 설정: {df.shape[0]:,}×{df.shape[1]}")

    def rebuild(self, step: int, history: str | None = None) -> list[str]:
        """결합 데이터에 레시피를 다시 적용해 작업 데이터를 만든다. 실패한 연산은 제외하고 경고를 돌려준다."""
        merged = self.p.load_df("merged")
        st = self.st
        good, warns = [], []
        df = merged
        for op in st["recipe"]:
            try:
                df = R.apply_row_op(df, op) if op["op"] in R.ROW_OPS else F.apply_op(df, op)
                good.append(op)
            except Exception as e:
                warns.append(f"{R.describe(op)} → 적용 불가로 제외: {e}")
        df = df.reset_index(drop=True)
        self.p.save_df("working", df)
        roles = st["roles"]
        kinds = {}
        for c in df.columns:
            kinds[c] = dataio.infer_kind(df[c], len(df))
        roles = {c: roles.get(c, dataio.default_role(kinds[c])) for c in df.columns}
        if st.get("target") in roles:
            roles[st["target"]] = "target"
        learned = [o for o in st["learned_ops"] if all(c in df.columns for c in o.get("columns", []))]
        self.update(recipe=good, roles=roles, kinds=kinds, working_shape=list(df.shape), learned_ops=learned)
        self.mark_change(step, history)
        return warns

    def working(self) -> pd.DataFrame:
        if not self.p.has_df("working"):
            self.merged()
            if not self.p.has_df("working"):
                self.rebuild(1)
        return self.cached("working", lambda: self.p.load_df("working"))

    def view(self) -> pd.DataFrame:
        """학습형 연산까지 전체 데이터로 적용한 탐색용 데이터."""
        learned = self.st["learned_ops"]
        if not learned:
            return self.working()
        return self.cached("view", lambda: R.view(self.working(), learned))

    def modeling_base(self) -> pd.DataFrame:
        return self.working()

    # ------------------------------------------------ 역할
    def features(self) -> list[str]:
        st = self.st
        tgt = st.get("target")
        return [c for c, r in st["roles"].items() if r == "feature" and c != tgt]

    def columns_meta(self) -> list[dict]:
        st = self.st
        try:
            df = self.working()
        except Exception:
            return []
        out = []
        for c in df.columns:
            s = df[c]
            out.append({"name": c, "kind": st["kinds"].get(c, "numeric"), "role": st["roles"].get(c, "feature"),
                        "numeric": bool(pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)),
                        "missing": int(s.isna().sum()), "nunique": int(s.nunique())})
        return out

    # ------------------------------------------------ 분할 데이터
    def split_data(self):
        sp = self.st.get("split")
        if not sp:
            raise ValueError("먼저 8단계에서 데이터를 분할하세요.")
        idx = self.p.load_obj("split_idx")
        base = self.modeling_base()
        tr, te = idx["train"], idx["test"]
        return base, tr, te, sp

    def xy(self):
        base, tr, te, sp = self.split_data()
        feats = sp["features"]
        tgt = self.st["target"]
        Xtr, Xte = base.loc[tr, feats], base.loc[te, feats]
        ytr, yte = base.loc[tr, tgt], base.loc[te, tgt]
        if self.st["target_type"] != "regression":
            cmap = {c: i for i, c in enumerate(self.st["classes"])}
            ytr = ytr.astype(str).map(cmap).astype(int)
            yte = yte.astype(str).map(cmap).astype(int)
        else:
            ytr, yte = pd.to_numeric(ytr), pd.to_numeric(yte)
        groups = base.loc[tr, sp["group_col"]].astype(str).values if sp.get("method") == "group" and sp.get("group_col") else None
        return Xtr, Xte, ytr, yte, groups

    # ------------------------------------------------ 입력 열(파생변수의 원천)
    def base_columns(self, feats: list[str]) -> list[str]:
        merged = self.merged()
        smp = merged.head(50)
        created = {}
        df = smp
        for i, op in enumerate(self.st["recipe"]):
            if op["op"] in R.ROW_OPS:
                continue
            before = set(df.columns)
            try:
                df = F.apply_op(df, op)
            except Exception:
                continue
            for c in df.columns:
                if c not in before:
                    created[c] = op
        need = []

        def resolve(c, depth=0):
            if depth > 20:
                return
            if c in created:
                op = created[c]
                srcs = [op.get(k) for k in ("source", "left", "right", "group") if op.get(k)]
                if op["op"] == "expr":
                    srcs += [m for m in df.columns if re.search(rf"(?<![\w가-힣]){re.escape(m)}(?![\w가-힣])", op["expr"])]
                for s in srcs:
                    if s != c:
                        resolve(s, depth + 1)
            elif c in merged.columns and c not in need:
                need.append(c)
        for f in feats:
            resolve(f)
        return need

    # ------------------------------------------------ 요약
    def summary(self) -> dict:
        st = self.st
        meta = self.p.meta
        results = {}
        changes = {int(k): v for k, v in st.get("changes", {}).items()}
        for key, m in st["results_meta"].items():
            s = step_of(key)
            stale = any(k < s and t > m["time"] for k, t in changes.items())
            results[key] = {"time": m["time"], "stale": stale, "step": s}
        steps = {}
        for s in STEP_TITLES:
            keys = [k for k, v in results.items() if v["step"] == s]
            steps[s] = {"title": STEP_TITLES[s], "done": bool(keys), "stale": any(results[k]["stale"] for k in keys)}
        from .analysis.recipe import describe
        return {
            "id": self.pid, "name": meta.get("name"), "sources": st["sources"], "merge": st["merge"],
            "working_shape": st["working_shape"], "columns": self.columns_meta() if st["sources"] else [],
            "target": st["target"], "target_type": st["target_type"], "classes": st.get("classes"),
            "recipe": [{"index": i, "text": describe(op), "op": op["op"]} for i, op in enumerate(st["recipe"])],
            "learned_ops": [{"index": i, "text": describe(op), "op": op["op"]} for i, op in enumerate(st["learned_ops"])],
            "split": {k: v for k, v in (st.get("split") or {}).items()} if st.get("split") else None,
            "preprocess": st.get("preprocess"), "models": st.get("models"), "best_model": st.get("best_model"),
            "results": results, "steps": steps,
            "history": st["history"][-30:], "report_items": st.get("report_items", []),
            "newdata": st.get("newdata", []),
        }
