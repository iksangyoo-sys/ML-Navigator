import { useMemo, useState } from 'react'
import Plot from './Plot'
import { downloadText, fmt, tableToCsv } from '../api'

// **굵게**, `코드` 정도만 지원하는 간단한 마크다운
export function Md({ text }) {
  if (!text) return null
  const lines = String(text).split('\n')
  return (
    <>
      {lines.map((line, i) => (
        <span key={i}>
          {inline(line)}
          {i < lines.length - 1 && <br />}
        </span>
      ))}
    </>
  )
}

function inline(s) {
  const parts = []
  const re = /(\*\*[^*]+\*\*|`[^`]+`)/g
  let last = 0
  let m
  let k = 0
  while ((m = re.exec(s))) {
    if (m.index > last) parts.push(s.slice(last, m.index))
    const t = m[0]
    parts.push(t.startsWith('**') ? <b key={k++}>{t.slice(2, -2)}</b> : <code key={k++}>{t.slice(1, -1)}</code>)
    last = m.index + t.length
  }
  if (last < s.length) parts.push(s.slice(last))
  return parts
}

function Table({ b }) {
  const [sort, setSort] = useState(null)
  const [q, setQ] = useState('')
  const rows = useMemo(() => {
    let r = b.rows
    if (q) {
      const ql = q.toLowerCase()
      r = r.filter((row) => row.some((v) => v != null && String(v).toLowerCase().includes(ql)))
    }
    if (sort) {
      const { i, dir } = sort
      r = [...r].sort((a, c) => {
        const x = a[i], y = c[i]
        if (x == null) return 1
        if (y == null) return -1
        if (typeof x === 'number' && typeof y === 'number') return (x - y) * dir
        return String(x).localeCompare(String(y), 'ko') * dir
      })
    }
    return r
  }, [b.rows, sort, q])
  const numeric = b.columns.map((_, i) => b.rows.slice(0, 20).every((r) => r[i] == null || typeof r[i] === 'number'))
  return (
    <div className="block-table">
      <div className="block-head">
        {b.title && <h4>{b.title}</h4>}
        <div className="table-tools">
          {b.rows.length > 10 && <input className="search" placeholder="표 내 검색" value={q} onChange={(e) => setQ(e.target.value)} />}
          <button
            className="ghost small"
            onClick={() => downloadText(b.download || `${(b.title || 'table').replace(/[\\/:*?"<>|`]/g, '_')}.csv`, tableToCsv(b.columns, b.rows))}
          >
            CSV
          </button>
        </div>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              {b.columns.map((c, i) => (
                <th
                  key={i}
                  className={numeric[i] ? 'num' : ''}
                  onClick={() => setSort((s) => (s && s.i === i ? { i, dir: -s.dir } : { i, dir: 1 }))}
                >
                  {c}
                  {sort?.i === i ? (sort.dir > 0 ? ' ▲' : ' ▼') : ''}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r, ri) => (
              <tr key={ri}>
                {r.map((v, ci) => (
                  <td key={ci} className={numeric[ci] ? 'num' : ''}>
                    {fmt(v)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {(b.truncated || b.caption) && (
        <p className="caption">
          {b.truncated && `전체 ${b.total.toLocaleString()}행 중 ${b.rows.length.toLocaleString()}행 표시. `}
          {b.caption}
        </p>
      )}
    </div>
  )
}

function Tabs({ b, pid }) {
  const [i, setI] = useState(0)
  return (
    <div className="block-tabs">
      <div className="tabbar">
        {b.tabs.map((t, k) => (
          <button key={k} className={k === i ? 'active' : ''} onClick={() => setI(k)}>
            {t.label}
          </button>
        ))}
      </div>
      <Blocks blocks={b.tabs[i]?.blocks || []} pid={pid} />
    </div>
  )
}

export default function Blocks({ blocks, pid }) {
  if (!blocks?.length) return null
  return (
    <div className="blocks">
      {blocks.map((b, i) => {
        switch (b.type) {
          case 'heading':
            return (
              <div key={i} className="block-heading">
                <h3>{b.title}</h3>
                {b.sub && <p>{<Md text={b.sub} />}</p>}
              </div>
            )
          case 'text':
            return <p key={i} className="block-text"><Md text={b.md} /></p>
          case 'note':
            return <div key={i} className={`note ${b.level}`}><Md text={b.text} /></div>
          case 'interpret':
            return (
              <div key={i} className="interpret">
                <div className="interpret-title">💡 {b.title}</div>
                <Md text={b.text} />
              </div>
            )
          case 'metrics':
            return (
              <div key={i} className="block-metrics">
                {b.title && <h4>{b.title}</h4>}
                <div className="metrics">
                  {b.items.map((m, k) => (
                    <div key={k} className="metric" title={m.hint || ''}>
                      <span>{m.label}</span>
                      <b>{m.value}</b>
                    </div>
                  ))}
                </div>
              </div>
            )
          case 'kv':
            return (
              <div key={i} className="block-kv">
                {b.title && <h4>{b.title}</h4>}
                <div className="kv">
                  {b.items.map((it, k) => (
                    <div key={k}><span>{it.k}</span><b>{it.v}</b></div>
                  ))}
                </div>
              </div>
            )
          case 'table':
            return <Table key={i} b={b} />
          case 'chart':
            return (
              <div key={i} className="block-chart">
                {b.title && <h4>{b.title}</h4>}
                <Plot figure={b.figure} />
                {b.caption && <p className="caption">{b.caption}</p>}
              </div>
            )
          case 'code':
            return (
              <div key={i} className="block-code">
                {b.title && <h4>{b.title}</h4>}
                <pre>{b.text}</pre>
              </div>
            )
          case 'tabs':
            return <Tabs key={i} b={b} pid={pid} />
          case 'download':
            return (
              <a key={i} className="button primary" href={`/api/projects/${pid}/${b.href}`}>
                ⬇ {b.label}
              </a>
            )
          default:
            return null
        }
      })}
    </div>
  )
}
