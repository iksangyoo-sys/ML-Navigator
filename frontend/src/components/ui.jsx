import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../api'
import Blocks from './Blocks'

export const ProjectCtx = createContext(null)
export const useProject = () => useContext(ProjectCtx)

// ---------------------------------------------------------------- 입력 컨트롤
export function Field({ label, hint, children, wide }) {
  return (
    <label className={'field' + (wide ? ' wide' : '')}>
      <span className="field-label">{label}</span>
      {children}
      {hint && <span className="field-hint">{hint}</span>}
    </label>
  )
}

export function Select({ value, onChange, options, placeholder, allowEmpty }) {
  const opts = options.map((o) => (typeof o === 'object' ? o : { value: o, label: String(o) }))
  return (
    <select value={value ?? ''} onChange={(e) => onChange(e.target.value)}>
      {(allowEmpty || placeholder) && <option value="">{placeholder || '(선택 안 함)'}</option>}
      {opts.map((o) => (
        <option key={String(o.value)} value={o.value}>
          {o.label}
        </option>
      ))}
    </select>
  )
}

export function Num({ value, onChange, min, max, step = 'any', placeholder }) {
  return (
    <input
      type="number"
      value={value ?? ''}
      min={min}
      max={max}
      step={step}
      placeholder={placeholder}
      onChange={(e) => onChange(e.target.value === '' ? '' : Number(e.target.value))}
    />
  )
}

export function Text({ value, onChange, placeholder }) {
  return <input type="text" value={value ?? ''} placeholder={placeholder} onChange={(e) => onChange(e.target.value)} />
}

export function Radio({ value, onChange, options }) {
  const opts = options.map((o) => (typeof o === 'object' ? o : { value: o, label: o }))
  return (
    <div className="radio">
      {opts.map((o) => (
        <button type="button" key={o.value} className={value === o.value ? 'on' : ''} onClick={() => onChange(o.value)}>
          {o.label}
        </button>
      ))}
    </div>
  )
}

export function Check({ checked, onChange, label }) {
  return (
    <label className="check">
      <input type="checkbox" checked={!!checked} onChange={(e) => onChange(e.target.checked)} /> {label}
    </label>
  )
}

export function MultiSelect({ value = [], onChange, options, height = 180, labels }) {
  const [q, setQ] = useState('')
  const opts = options.filter((o) => !q || o.toLowerCase().includes(q.toLowerCase()))
  const set = new Set(value)
  const toggle = (o) => onChange(set.has(o) ? value.filter((v) => v !== o) : [...value, o])
  return (
    <div className="multi">
      <div className="multi-tools">
        <input placeholder="검색" value={q} onChange={(e) => setQ(e.target.value)} />
        <button type="button" className="ghost small" onClick={() => onChange([...new Set([...value, ...opts])])}>전체</button>
        <button type="button" className="ghost small" onClick={() => onChange(value.filter((v) => !opts.includes(v)))}>해제</button>
        <span className="muted small">{value.length}/{options.length}</span>
      </div>
      <div className="multi-list" style={{ maxHeight: height }}>
        {opts.map((o) => (
          <label key={o} className={set.has(o) ? 'on' : ''}>
            <input type="checkbox" checked={set.has(o)} onChange={() => toggle(o)} />
            <span>{o}</span>
            {labels?.[o] && <em>{labels[o]}</em>}
          </label>
        ))}
        {!opts.length && <span className="muted small">항목 없음</span>}
      </div>
    </div>
  )
}

export function Section({ title, desc, children, actions }) {
  return (
    <section className="card">
      {(title || actions) && (
        <div className="card-head">
          <div>
            {title && <h3>{title}</h3>}
            {desc && <p className="muted">{desc}</p>}
          </div>
          {actions && <div className="card-actions">{actions}</div>}
        </div>
      )}
      {children}
    </section>
  )
}

export function Grid({ children, cols = 3 }) {
  return <div className={`grid g${cols}`}>{children}</div>
}

// ---------------------------------------------------------------- 작업 실행 훅
export function useRun() {
  const { pid, refresh, setResult } = useProject()
  const [job, setJob] = useState(null)
  const [error, setError] = useState(null)
  const timer = useRef(null)
  useEffect(() => () => clearTimeout(timer.current), [])
  const run = useCallback(
    async (name, params = {}, opts = {}) => {
      setError(null)
      try {
        const { job: id } = await api.post(`/projects/${pid}/run/${name}`, params)
        setJob({ id, status: 'queued', pct: 0, message: '대기 중', action: name })
        const poll = async () => {
          try {
            const j = await api.get(`/jobs/${id}`)
            setJob(j)
            if (j.status === 'done') {
              setResult(j.key, { blocks: j.blocks, time: Date.now() / 1000 })
              await refresh()
              opts.onDone?.(j)
              setTimeout(() => setJob(null), 400)
            } else if (j.status === 'failed') {
              setError(j.error)
              setJob(null)
            } else if (j.status === 'cancelled') {
              setError('작업을 취소했습니다.')
              setJob(null)
            } else timer.current = setTimeout(poll, 450)
          } catch (e) {
            setError(e.message)
            setJob(null)
          }
        }
        timer.current = setTimeout(poll, 250)
      } catch (e) {
        setError(e.message)
      }
    },
    [pid, refresh, setResult],
  )
  const cancel = useCallback(async () => {
    if (job?.id) await api.post(`/jobs/${job.id}/cancel`)
  }, [job])
  return { run, job, running: !!job, error, setError, cancel }
}

export function RunStatus({ r }) {
  if (r.job)
    return (
      <div className="runstatus">
        <div className="bar"><div style={{ width: `${Math.round((r.job.pct || 0) * 100)}%` }} /></div>
        <span>{r.job.message || '실행 중'}</span>
        <button className="ghost small" onClick={r.cancel}>취소</button>
      </div>
    )
  if (r.error)
    return (
      <div className="note warn">
        {r.error} <button className="ghost small" onClick={() => r.setError(null)}>닫기</button>
      </div>
    )
  return null
}

export function RunButton({ r, name, params, children, primary = true, disabled, onDone }) {
  return (
    <button className={primary ? 'primary' : ''} disabled={r.running || disabled} onClick={() => r.run(name, typeof params === 'function' ? params() : params, { onDone })}>
      {r.running && r.job?.action === name ? '실행 중…' : children}
    </button>
  )
}

// ---------------------------------------------------------------- 결과 패널
export function Result({ rkey, title, empty }) {
  const { results, summary, pid, auth } = useProject()
  const res = results[rkey]
  const meta = summary?.results?.[rkey]
  const [ai, setAi] = useState(null)
  const [aiBusy, setAiBusy] = useState(false)
  if (!res) return empty ? <div className="empty">{empty}</div> : null
  const askAi = async () => {
    setAiBusy(true)
    try {
      const r = await api.post(`/projects/${pid}/ai/explain`, { key: rkey })
      setAi(r.text)
    } catch (e) {
      setAi('⚠ ' + e.message)
    }
    setAiBusy(false)
  }
  return (
    <section className="card result">
      <div className="result-head">
        <h3>{title || '결과'}</h3>
        <div className="result-meta">
          {meta?.stale && <span className="badge warn" title="앞 단계가 바뀌었습니다. 다시 실행하세요.">재실행 필요</span>}
          {res.time && <span className="muted small">{new Date(res.time * 1000).toLocaleString('ko-KR')}</span>}
          {auth?.ai && (
            <button className="ghost small" onClick={askAi} disabled={aiBusy}>
              {aiBusy ? 'AI 해설 생성 중…' : 'AI 해설'}
            </button>
          )}
        </div>
      </div>
      {ai && (
        <div className="interpret ai">
          <div className="interpret-title">🤖 AI 해설 (초안, 검토 후 사용)</div>
          <div style={{ whiteSpace: 'pre-wrap' }}>{ai}</div>
        </div>
      )}
      <Blocks blocks={res.blocks} pid={pid} />
    </section>
  )
}

// ---------------------------------------------------------------- 열 도우미
export function useColumns() {
  const { summary } = useProject()
  return useMemo(() => {
    const cols = summary?.columns || []
    const target = summary?.target
    const all = cols.map((c) => c.name)
    const numeric = cols.filter((c) => c.numeric).map((c) => c.name)
    const categorical = cols.filter((c) => !c.numeric || c.nunique <= 10).map((c) => c.name)
    const features = cols.filter((c) => c.role === 'feature' && c.name !== target).map((c) => c.name)
    const numFeatures = features.filter((c) => numeric.includes(c))
    const catFeatures = features.filter((c) => !numeric.includes(c))
    const missing = cols.filter((c) => c.missing > 0).map((c) => c.name)
    return { cols, all, numeric, categorical, features, numFeatures, catFeatures, missing, target }
  }, [summary])
}

export function Need({ when, children }) {
  return when ? <div className="note warn">{children}</div> : null
}
