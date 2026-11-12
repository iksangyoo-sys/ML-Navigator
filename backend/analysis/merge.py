"""2단계: 행·열 방향 결합과 결합 품질 진단."""
from __future__ import annotations

import pandas as pd

from . import blocks as B


def common_columns(dfs: list[pd.DataFrame]) -> list[str]:
    if not dfs:
        return []
    cols = list(dfs[0].columns)
    for d in dfs[1:]:
        cols = [c for c in cols if c in d.columns]
    return cols


def key_relation(left: pd.DataFrame, right: pd.DataFrame, keys: list[str]) -> str:
    lu = not left.duplicated(keys).any()
    ru = not right.duplicated(keys).any()
    return {(True, True): "1:1", (False, True): "N:1", (True, False): "1:N", (False, False): "N:N"}[(lu, ru)]


def concat_rows(named: list[tuple[str, pd.DataFrame]]) -> tuple[pd.DataFrame, list]:
    all_cols = []
    for _, d in named:
        for c in d.columns:
            if c not in all_cols:
                all_cols.append(c)
    rows = []
    for name, d in named:
        missing = [c for c in all_cols if c not in d.columns]
        rows.append({"데이터": name, "행": len(d), "열": d.shape[1], "없는 열": ", ".join(missing) or "-"})
    # 타입 불일치 점검
    mism = []
    for c in all_cols:
        types = {name: str(d[c].dtype) for name, d in named if c in d.columns}
        kinds = {("num" if t.startswith(("int", "float")) else t) for t in types.values()}
        if len(kinds) > 1:
            mism.append({"열": c, **types})
    merged = pd.concat([d.assign(**{"__source__": name}) for name, d in named], axis=0, ignore_index=True)
    merged = merged.rename(columns={"__source__": "원본파일"})
    log = pd.DataFrame(rows)
    log.loc[len(log)] = {"데이터": "결합 후", "행": len(merged), "열": merged.shape[1], "없는 열": "-"}
    out = [B.table(log, "행 방향 결합 로그"),
           B.note("결합 후 어떤 파일에서 온 행인지 `원본파일` 열로 구분합니다. 필요 없으면 열 역할에서 제외하세요.")]
    if mism:
        out.append(B.table(pd.DataFrame(mism), "파일 간 타입이 다른 열"))
        out.append(B.note("같은 이름인데 타입이 다른 열이 있습니다. 숫자가 문자로 읽혔는지 확인하세요.", "warn"))
    if any(r["없는 열"] != "-" for r in rows):
        out.append(B.note("일부 파일에 없는 열은 결측으로 채워졌습니다.", "warn"))
    return merged, out


def merge_columns(named: list[tuple[str, pd.DataFrame]], keys_list: list[list[str]], how: str) -> tuple[pd.DataFrame, list]:
    merged = named[0][1].copy()
    logs = []
    diag_blocks = []
    for i in range(1, len(named)):
        name, right = named[i]
        keys = keys_list[i - 1] if i - 1 < len(keys_list) and keys_list[i - 1] else \
            [c for c in merged.columns if c in right.columns]
        if not keys:
            raise ValueError(f"{name}과(와) 결합할 공통 키가 없습니다.")
        for k in keys:
            if k not in merged.columns or k not in right.columns:
                raise ValueError(f"키 `{k}`가 양쪽에 모두 있어야 합니다.")
        rel = key_relation(merged, right, keys)
        ind = pd.merge(merged[keys].drop_duplicates(), right[keys].drop_duplicates(), on=keys, how="outer", indicator=True)
        left_only = int((ind["_merge"] == "left_only").sum())
        right_only = int((ind["_merge"] == "right_only").sum())
        both = int((ind["_merge"] == "both").sum())
        before = merged.shape
        overlap = [c for c in right.columns if c in merged.columns and c not in keys]
        merged = pd.merge(merged, right, on=keys, how=how, suffixes=("", f"_{i+1}"))
        growth = merged.shape[0] / max(before[0], 1)
        logs.append({"단계": f"결합 {i}", "오른쪽 데이터": name, "결합 키": ", ".join(keys), "키 관계": rel,
                     "방식": how, "왼쪽 크기": f"{before[0]:,}×{before[1]}", "오른쪽 크기": f"{right.shape[0]:,}×{right.shape[1]}",
                     "결합 후 크기": f"{merged.shape[0]:,}×{merged.shape[1]}", "행 증가율": round(growth, 3),
                     "키 매칭(양쪽)": both, "왼쪽만": left_only, "오른쪽만": right_only})
        if rel == "N:N":
            diag_blocks.append(B.note(f"결합 {i}: 키 `{', '.join(keys)}`가 양쪽 모두 중복되어 N:N 결합입니다. 행이 곱으로 늘어날 수 있으니 키를 확인하세요.", "warn"))
        if growth > 1.0001 and how in ("left", "inner"):
            diag_blocks.append(B.note(f"결합 {i}: 행이 {before[0]:,} → {merged.shape[0]:,}로 늘었습니다(행 증식). 오른쪽 키가 고유한지 확인하세요.", "warn"))
        if left_only or right_only:
            diag_blocks.append(B.note(f"결합 {i}: 키가 한쪽에만 있는 조합이 왼쪽 {left_only:,}개, 오른쪽 {right_only:,}개 있습니다. "
                                      f"{'inner 결합에서는 이 행들이 빠집니다.' if how == 'inner' else '매칭되지 않은 행의 오른쪽 열은 결측이 됩니다.'}", "info"))
        if overlap:
            diag_blocks.append(B.note(f"결합 {i}: 키가 아닌 같은 이름의 열 {', '.join(overlap)}은 `_{i+1}` 접미사를 붙여 구분했습니다.", "info"))
    return merged, [B.table(pd.DataFrame(logs), "열 방향 결합 로그")] + diag_blocks


def key_candidates(named: list[tuple[str, pd.DataFrame]]) -> list[dict]:
    """공통 열마다 고유성·매칭률을 계산해 결합 키 후보를 제안."""
    if len(named) < 2:
        return []
    left = named[0][1]
    out = []
    for name, right in named[1:]:
        for c in [c for c in left.columns if c in right.columns]:
            lu = left[c].nunique() / max(len(left), 1)
            ru = right[c].nunique() / max(len(right), 1)
            lv = set(left[c].dropna().astype(str).unique()[:50000])
            rv = set(right[c].dropna().astype(str).unique()[:50000])
            match = len(lv & rv) / max(len(lv), 1)
            out.append({"오른쪽 데이터": name, "공통 열": c, "왼쪽 고유비율": round(lu, 3), "오른쪽 고유비율": round(ru, 3),
                        "왼쪽 값 매칭률": round(match, 3)})
    return out
