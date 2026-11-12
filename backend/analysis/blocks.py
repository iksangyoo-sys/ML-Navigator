"""화면에 표시할 결과 블록 생성기.

모든 분석 함수는 블록 리스트를 반환하고, 프런트엔드는 블록 타입에 따라 범용 렌더링한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..storage import to_jsonable

MAX_TABLE_ROWS = 500


def text(md: str) -> dict:
    return {"type": "text", "md": md}


def note(msg: str, level: str = "info") -> dict:
    return {"type": "note", "level": level, "text": msg}


def interpret(msg: str, title: str = "해석 가이드") -> dict:
    return {"type": "interpret", "title": title, "text": msg}


def heading(title: str, sub: str | None = None) -> dict:
    return {"type": "heading", "title": title, "sub": sub}


def code(content: str, title: str | None = None) -> dict:
    return {"type": "code", "title": title, "text": content}


def metrics(items: list[tuple], title: str | None = None) -> dict:
    """items: (label, value[, hint])"""
    out = []
    for it in items:
        label, value = it[0], it[1]
        hint = it[2] if len(it) > 2 else None
        out.append({"label": label, "value": fmt(value), "hint": hint})
    return {"type": "metrics", "title": title, "items": out}


def fmt(v, digits: int = 4):
    if v is None:
        return "-"
    if isinstance(v, (bool, np.bool_)):
        return "예" if v else "아니오"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        if np.isnan(v) or np.isinf(v):
            return "-"
        a = abs(float(v))
        if a != 0 and (a < 1e-3 or a >= 1e7):
            return f"{float(v):.3e}"
        if a >= 1000:
            return f"{float(v):,.2f}"
        return f"{float(v):.{digits}g}" if a < 1 else f"{float(v):.{digits}f}".rstrip("0").rstrip(".")
    return str(v)


def table(df: pd.DataFrame, title: str | None = None, index: bool = False, max_rows: int = MAX_TABLE_ROWS,
          download: str | None = None, highlight: dict | None = None, caption: str | None = None) -> dict:
    d = df.copy()
    if index:
        d = d.reset_index()
    total = len(d)
    if total > max_rows:
        d = d.head(max_rows)
    cols = [str(c) for c in d.columns]
    rows = []
    for rec in d.itertuples(index=False):
        rows.append([cell(v) for v in rec])
    out = {"type": "table", "title": title, "columns": cols, "rows": rows, "total": total,
           "truncated": total > max_rows, "caption": caption}
    if download:
        out["download"] = download
    if highlight:
        out["highlight"] = highlight
    return out


def cell(v):
    if isinstance(v, (float, np.floating)):
        if np.isnan(v) or np.isinf(v):
            return None
        return round(float(v), 6)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(v, (list, tuple, dict)):
        return str(v)
    return v if isinstance(v, (int, float, str)) else str(v)


def chart(fig, title: str | None = None, height: int | None = None, caption: str | None = None) -> dict:
    """plotly Figure 또는 dict를 차트 블록으로."""
    spec = fig.to_plotly_json() if hasattr(fig, "to_plotly_json") else fig
    spec = to_jsonable(spec)
    layout = spec.setdefault("layout", {})
    layout.pop("template", None)
    if height:
        layout["height"] = height
    if title and not layout.get("title"):
        pass
    return {"type": "chart", "title": title, "figure": {"data": spec.get("data", []), "layout": layout}, "caption": caption}


def tabs(items: list[tuple[str, list]]) -> dict:
    return {"type": "tabs", "tabs": [{"label": label, "blocks": blocks} for label, blocks in items]}


def kv(pairs: list[tuple], title: str | None = None) -> dict:
    return {"type": "kv", "title": title, "items": [{"k": k, "v": fmt(v) if not isinstance(v, str) else v} for k, v in pairs]}


def columns_info(df: pd.DataFrame) -> list[str]:
    return [str(c) for c in df.columns]
