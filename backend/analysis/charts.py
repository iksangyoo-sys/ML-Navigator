"""Plotly 차트 공통 스타일."""
from __future__ import annotations

import plotly.graph_objects as go

PALETTE = ["#2563eb", "#f97316", "#10b981", "#e11d48", "#8b5cf6", "#0ea5e9", "#eab308", "#64748b",
           "#14b8a6", "#db2777", "#84cc16", "#a855f7"]
FONT = "Pretendard, 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif"


def style(fig: go.Figure, title: str | None = None, x: str | None = None, y: str | None = None,
          height: int = 420, legend: bool = True) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0.01, "font": {"size": 15}} if title else None,
        font={"family": FONT, "size": 12, "color": "#1f2937"},
        colorway=PALETTE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        margin={"l": 60, "r": 24, "t": 48 if title else 24, "b": 52},
        height=height,
        showlegend=legend,
        legend={"orientation": "h", "y": -0.18, "x": 0},
        hoverlabel={"font": {"family": FONT}},
    )
    fig.update_xaxes(gridcolor="#eef2f7", zerolinecolor="#cbd5e1", linecolor="#cbd5e1", title_text=x)
    fig.update_yaxes(gridcolor="#eef2f7", zerolinecolor="#cbd5e1", linecolor="#cbd5e1", title_text=y, automargin=True)
    fig.update_xaxes(automargin=True)
    return fig
