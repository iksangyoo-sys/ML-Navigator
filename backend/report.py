"""보고서: 단계별 분석 보고서와 경영 요약을 HTML로 만든다(브라우저 인쇄로 PDF 저장)."""
from __future__ import annotations

import html
import json
import re
import time

from .workspace import STEP_TITLES, WS, step_of

CSS = """
:root{--ink:#0f172a;--muted:#475569;--line:#e2e8f0;--blue:#2563eb;--bg:#ffffff}
*{box-sizing:border-box}body{font-family:Pretendard,'Malgun Gothic',sans-serif;color:var(--ink);background:var(--bg);margin:0;line-height:1.6}
.page{max-width:1080px;margin:0 auto;padding:40px 32px}
h1{font-size:28px;margin:0 0 4px}h2{font-size:21px;margin:40px 0 12px;padding-bottom:6px;border-bottom:2px solid var(--ink)}
h3{font-size:16px;margin:24px 0 8px}.muted{color:var(--muted)}
table{border-collapse:collapse;width:100%;font-size:12px;margin:8px 0 16px}th,td{border:1px solid var(--line);padding:4px 8px;text-align:left}
th{background:#f1f5f9}td.num{text-align:right;font-variant-numeric:tabular-nums}
.metrics{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px;margin:12px 0}
.metric{border:1px solid var(--line);border-radius:8px;padding:8px 12px}.metric b{display:block;font-size:18px}.metric span{font-size:12px;color:var(--muted)}
.note{border-left:4px solid var(--blue);background:#eff6ff;padding:8px 12px;margin:8px 0;white-space:pre-wrap}
.note.warn{border-color:#f97316;background:#fff7ed}.note.success{border-color:#10b981;background:#ecfdf5}
.interp{border:1px solid #c7d2fe;background:#eef2ff;border-radius:8px;padding:10px 14px;margin:10px 0}
pre{background:#f8fafc;border:1px solid var(--line);padding:8px;font-size:11px;white-space:pre-wrap}
.chart{margin:8px 0 16px;page-break-inside:avoid}.toolbar{position:sticky;top:0;background:#fff;border-bottom:1px solid var(--line);padding:8px 32px;display:flex;gap:8px}
button{font:inherit;padding:6px 14px;border-radius:6px;border:1px solid var(--line);background:#fff;cursor:pointer}
.exec td:first-child{width:180px;background:#f8fafc;font-weight:600}
@media print{.toolbar{display:none}.page{padding:0}h2{page-break-before:auto}}
"""


def md(text: str) -> str:
    t = html.escape(text or "")
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"`(.+?)`", r"<code>\1</code>", t)
    return t.replace("\n", "<br>")


def fmt_cell(v):
    if v is None:
        return "", False
    if isinstance(v, bool):
        return ("예" if v else "아니오"), False
    if isinstance(v, (int, float)):
        if isinstance(v, float):
            a = abs(v)
            s = f"{v:.3e}" if a and (a < 1e-3 or a >= 1e7) else f"{v:,.4f}".rstrip("0").rstrip(".")
        else:
            s = f"{v:,}"
        return s, True
    return html.escape(str(v)), False


_chart_id = [0]


def render(blocks: list, max_rows: int = 60) -> str:
    out = []
    for b in blocks or []:
        t = b.get("type")
        if t == "heading":
            out.append(f"<h3>{md(b['title'])}</h3>" + (f"<p class='muted'>{md(b['sub'])}</p>" if b.get("sub") else ""))
        elif t == "text":
            out.append(f"<p>{md(b['md'])}</p>")
        elif t == "note":
            out.append(f"<div class='note {b.get('level', '')}'>{md(b['text'])}</div>")
        elif t == "interpret":
            out.append(f"<div class='interp'><b>{html.escape(b.get('title') or '해석')}</b><br>{md(b['text'])}</div>")
        elif t == "metrics":
            items = "".join(f"<div class='metric'><span>{html.escape(str(i['label']))}</span><b>{html.escape(str(i['value']))}</b></div>" for i in b["items"])
            out.append((f"<h3>{html.escape(b['title'])}</h3>" if b.get("title") else "") + f"<div class='metrics'>{items}</div>")
        elif t == "kv":
            rows = "".join(f"<tr><td>{html.escape(str(i['k']))}</td><td>{html.escape(str(i['v']))}</td></tr>" for i in b["items"])
            out.append((f"<h3>{html.escape(b['title'])}</h3>" if b.get("title") else "") + f"<table>{rows}</table>")
        elif t == "table":
            head = "".join(f"<th>{html.escape(c)}</th>" for c in b["columns"])
            body = []
            for r in b["rows"][:max_rows]:
                cells = []
                for v in r:
                    s, num = fmt_cell(v)
                    cells.append(f"<td class='{'num' if num else ''}'>{s}</td>")
                body.append("<tr>" + "".join(cells) + "</tr>")
            more = f"<p class='muted'>전체 {b['total']:,}행 중 {min(max_rows, len(b['rows']))}행 표시</p>" if b["total"] > max_rows else ""
            out.append((f"<h3>{html.escape(b['title'])}</h3>" if b.get("title") else "") + f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>{more}")
        elif t == "chart":
            _chart_id[0] += 1
            cid = f"c{_chart_id[0]}"
            fig = b["figure"]
            out.append(f"<div class='chart' id='{cid}'></div><script>Plotly.newPlot('{cid}',{json.dumps(fig['data'], ensure_ascii=False)},"
                       f"{json.dumps(fig['layout'], ensure_ascii=False)},{{displaylogo:false,responsive:true}});</script>")
            if b.get("caption"):
                out.append(f"<p class='muted'>{md(b['caption'])}</p>")
        elif t == "code":
            out.append((f"<h3>{html.escape(b['title'])}</h3>" if b.get("title") else "") + f"<pre>{html.escape(b['text'])}</pre>")
        elif t == "tabs":
            for tab in b["tabs"]:
                out.append(f"<h3>▸ {html.escape(tab['label'])}</h3>" + render(tab["blocks"], max_rows))
    return "\n".join(out)


def page(title: str, body: str) -> str:
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script><style>{CSS}</style></head>
<body><div class="toolbar"><button onclick="window.print()">PDF로 저장 / 인쇄</button></div><div class="page">{body}</div></body></html>"""


def step_report(ws: WS, keys: list[str] | None = None) -> str:
    st = ws.st
    meta = ws.p.meta
    results = ws.p.all_results()
    keys = keys or sorted(results, key=lambda k: (step_of(k), k))
    parts = [f"<h1>{html.escape(meta.get('name', ''))} 단계별 분석 보고서</h1>",
             f"<p class='muted'>작성 {time.strftime('%Y-%m-%d %H:%M')} · 데이터 {st.get('working_shape')} · 종속변수 {st.get('target') or '-'} · "
             "학습형 전처리는 학습 데이터로만 적합(누수 방지)</p>"]
    src = "".join(f"<tr><td>{html.escape(s['name'])}</td><td>{s.get('rows', '')}</td><td>{s.get('cols', '')}</td><td>{s.get('encoding') or s.get('format', '')}</td><td>{s.get('hash', '')}</td></tr>" for s in st["sources"])
    parts.append(f"<h2>데이터 출처</h2><table><tr><th>파일</th><th>행</th><th>열</th><th>형식/인코딩</th><th>해시</th></tr>{src}</table>")
    from .analysis.recipe import describe
    if st["recipe"] or st["learned_ops"]:
        rows = "".join(f"<tr><td>결정적</td><td>{html.escape(describe(o))}</td></tr>" for o in st["recipe"])
        rows += "".join(f"<tr><td>학습형</td><td>{html.escape(describe(o))}</td></tr>" for o in st["learned_ops"])
        parts.append(f"<h2>전처리 레시피</h2><table><tr><th>구분</th><th>연산</th></tr>{rows}</table>")
    cur = None
    for k in keys:
        if k not in results:
            continue
        s = step_of(k)
        if s != cur:
            parts.append(f"<h2>{s}. {html.escape(STEP_TITLES.get(s, ''))}</h2>")
            cur = s
        parts.append(render(results[k]["blocks"]))
    return page(f"{meta.get('name')} 분석 보고서", "\n".join(parts))


def executive(ws: WS, fields: dict) -> str:
    st = ws.st
    meta = ws.p.meta
    best = next((m for m in st.get("models") or [] if m["key"] == st.get("best_model")), None)
    base = next((m for m in st.get("models") or [] if m["key"] == "baseline"), None)
    kpi_rows = ""
    if best:
        for k, v in list(best["test"].items())[:6]:
            bv = base["test"].get(k) if base else None
            kpi_rows += f"<tr><td>{html.escape(k)}</td><td class='num'>{fmt_cell(bv)[0]}</td><td class='num'>{fmt_cell(v)[0]}</td></tr>"
    rows = [("의사결정 질문", fields.get("question")), ("핵심 결론", fields.get("conclusion")), ("기준 대비 성과", fields.get("kpi")),
            ("불확실성·한계", fields.get("uncertainty")), ("권고 사항", fields.get("recommendation")), ("담당", fields.get("owner")), ("기한", fields.get("due"))]
    tbl = "".join(f"<tr><td>{a}</td><td>{md(b or '')}</td></tr>" for a, b in rows)
    body = [f"<h1>{html.escape(meta.get('name', ''))} 경영 요약</h1>",
            f"<p class='muted'>{time.strftime('%Y-%m-%d')} · 종속변수 {html.escape(str(st.get('target')))} · 최적 모델 {html.escape(best['name']) if best else '-'}</p>",
            f"<table class='exec'>{tbl}</table>"]
    if kpi_rows:
        body.append(f"<h2>모델 성능 (평가 데이터)</h2><table><tr><th>지표</th><th>기준선</th><th>최적 모델</th></tr>{kpi_rows}</table>")
    results = ws.p.all_results()
    for key, title in [("s14_batch", "조정 변수별 인과효과"), ("s15_opt", "목표 달성 최적 조건"), ("s15", "개선 시나리오(Counterfactual)"), ("s11_explain", "핵심 영향 변수")]:
        if key in results:
            bl = [b for b in results[key]["blocks"] if b["type"] in ("chart", "table", "interpret", "metrics")][:4]
            if key == "s11_explain":
                tabs = [b for b in results[key]["blocks"] if b["type"] == "tabs"]
                bl = tabs[0]["tabs"][0]["blocks"][:1] if tabs else []
            body.append(f"<h2>{title}</h2>" + render(bl, 15))
    return page(f"{meta.get('name')} 경영 요약", "\n".join(body))
