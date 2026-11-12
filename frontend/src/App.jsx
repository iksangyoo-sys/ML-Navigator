import { useCallback, useEffect, useState } from 'react'
import { api } from './api'
import { ProjectCtx } from './components/ui'
import { STEPS } from './steps'

function Login({ onOk }) {
  const [pw, setPw] = useState('')
  const [err, setErr] = useState('')
  const submit = async (e) => {
    e.preventDefault()
    try {
      await api.post('/login', { password: pw })
      onOk()
    } catch (e) {
      setErr(e.message)
    }
  }
  return (
    <div className="login">
      <form onSubmit={submit} className="card">
        <div className="logo big">◆</div>
        <h2>빅데이터 분석 자동화</h2>
        <p className="muted">비밀번호를 입력하세요.</p>
        <input type="password" autoFocus value={pw} onChange={(e) => setPw(e.target.value)} placeholder="비밀번호" />
        {err && <div className="note warn">{err}</div>}
        <button className="primary">로그인</button>
      </form>
    </div>
  )
}

function Projects({ onOpen, auth }) {
  const [list, setList] = useState(null)
  const [name, setName] = useState('')
  const load = () => api.get('/projects').then(setList)
  useEffect(() => {
    load()
  }, [])
  const create = async () => {
    const { id } = await api.post('/projects', { name: name || '새 분석 프로젝트' })
    onOpen(id)
  }
  const remove = async (p) => {
    if (!window.confirm(`'${p.name}' 프로젝트를 삭제할까요? 데이터와 모델이 모두 지워집니다.`)) return
    await api.del(`/projects/${p.id}`)
    load()
  }
  return (
    <div className="projects">
      <header className="topbar">
        <div className="brand"><span className="logo">◆</span> 빅데이터 분석 자동화</div>
        {auth?.required && <button className="ghost" onClick={() => api.post('/logout').then(() => location.reload())}>로그아웃</button>}
      </header>
      <main className="projects-main">
        <section className="hero">
          <h1>데이터가 바뀌어도, 분석 흐름은 그대로</h1>
          <p>불러오기부터 결합·파생변수·결측·이상치·EDA·모델링·설명 가능성·DoWhy 인과분석·Counterfactual·신규 데이터 예측까지 16단계를 순서대로 진행합니다.</p>
          <div className="create">
            <input placeholder="프로젝트 이름 (예: 양극재 탭밀도 예측)" value={name} onChange={(e) => setName(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && create()} />
            <button className="primary" onClick={create}>새 프로젝트</button>
          </div>
        </section>
        <h3>프로젝트</h3>
        {!list ? (
          <p className="muted">불러오는 중…</p>
        ) : !list.length ? (
          <p className="muted">아직 프로젝트가 없습니다.</p>
        ) : (
          <div className="project-grid">
            {list.map((p) => (
              <div key={p.id} className="card project-card" onClick={() => onOpen(p.id)}>
                <h4>{p.name}</h4>
                <p className="muted small">
                  {p.shape ? `${p.shape[0].toLocaleString()}행 × ${p.shape[1]}열` : '데이터 없음'} · 타깃 {p.target || '-'}
                </p>
                <div className="progress-dots">
                  <span>결과 {p.steps_done}개</span>
                  <span className="muted small">{new Date(p.updated * 1000).toLocaleString('ko-KR')}</span>
                </div>
                <button className="ghost small danger" onClick={(e) => { e.stopPropagation(); remove(p) }}>삭제</button>
              </div>
            ))}
          </div>
        )}
      </main>
    </div>
  )
}

function Workspace({ pid, onBack, auth }) {
  const [summary, setSummary] = useState(null)
  const [results, setResults] = useState({})
  const [step, setStep] = useState(() => Number(sessionStorage.getItem(`step_${pid}`)) || 1)
  const [err, setErr] = useState(null)
  const refresh = useCallback(async () => {
    try {
      setSummary(await api.get(`/projects/${pid}`))
    } catch (e) {
      setErr(e.message)
    }
  }, [pid])
  const setResult = useCallback((k, v) => setResults((r) => ({ ...r, [k]: v })), [])
  useEffect(() => {
    refresh()
  }, [refresh])
  useEffect(() => {
    sessionStorage.setItem(`step_${pid}`, step)
    if (step === 'report') return
    api.get(`/projects/${pid}/results?step=${step}`).then((r) => setResults((old) => ({ ...old, ...Object.fromEntries(Object.entries(r).filter(([, v]) => v)) })))
    window.scrollTo(0, 0)
  }, [pid, step])
  if (err) return <div className="note warn">{err}</div>
  if (!summary) return <div className="loading">불러오는 중…</div>
  const cur = STEPS.find((s) => s.id === step) || STEPS[0]
  const Comp = cur.comp
  const rename = async () => {
    const n = window.prompt('프로젝트 이름', summary.name)
    if (n) setSummary(await api.patch(`/projects/${pid}`, { name: n }))
  }
  const idx = STEPS.findIndex((s) => s.id === cur.id)
  return (
    <ProjectCtx.Provider value={{ pid, summary, refresh, results, setResult, auth, setStep }}>
      <div className="ws">
        <aside className="side">
          <div className="side-brand" onClick={onBack} title="프로젝트 목록">
            <span className="logo">◆</span> 분석 자동화
          </div>
          <div className="side-project" onClick={rename} title="이름 바꾸기">{summary.name}</div>
          <nav>
            {STEPS.map((s) => {
              const st = typeof s.id === 'number' ? summary.steps[s.id] : null
              return (
                <button key={s.id} className={'step' + (s.id === cur.id ? ' active' : '')} onClick={() => setStep(s.id)}>
                  <span className={'dot' + (st?.stale ? ' stale' : st?.done ? ' done' : '')}>{s.num}</span>
                  <span className="step-title">{s.title}</span>
                </button>
              )
            })}
          </nav>
        </aside>
        <div className="main">
          <header className="ws-head">
            <div>
              <div className="crumb">{cur.num !== '★' ? `${cur.num}단계` : '보고서'}</div>
              <h1>{cur.title}</h1>
              <p className="muted">{cur.desc}</p>
            </div>
            <div className="ws-status">
              <div className="chip">{summary.working_shape ? `${summary.working_shape[0].toLocaleString()}행 × ${summary.working_shape[1]}열` : '데이터 없음'}</div>
              <div className="chip">타깃 {summary.target || '-'}</div>
            </div>
          </header>
          <div className="content">
            <Comp key={cur.id} />
            <div className="pager">
              {idx > 0 && <button className="ghost" onClick={() => setStep(STEPS[idx - 1].id)}>← {STEPS[idx - 1].title}</button>}
              <span />
              {idx < STEPS.length - 1 && <button onClick={() => setStep(STEPS[idx + 1].id)}>{STEPS[idx + 1].title} →</button>}
            </div>
          </div>
        </div>
      </div>
    </ProjectCtx.Provider>
  )
}

export default function App() {
  const [auth, setAuth] = useState(null)
  const [pid, setPid] = useState(() => new URLSearchParams(location.search).get('p'))
  const check = () => api.get('/auth').then(setAuth).catch(() => setAuth({ required: true, authed: false }))
  useEffect(() => {
    check()
  }, [])
  useEffect(() => {
    const u = new URL(location.href)
    if (pid) u.searchParams.set('p', pid)
    else u.searchParams.delete('p')
    history.replaceState(null, '', u)
  }, [pid])
  if (!auth) return <div className="loading">불러오는 중…</div>
  if (auth.required && !auth.authed) return <Login onOk={check} />
  return pid ? <Workspace pid={pid} onBack={() => setPid(null)} auth={auth} /> : <Projects onOpen={setPid} auth={auth} />
}
