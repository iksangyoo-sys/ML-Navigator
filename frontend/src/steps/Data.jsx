import { useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../api'
import { Check, Field, Grid, MultiSelect, Need, Num, Radio, Result, RunButton, RunStatus, Section, Select, Text, useColumns, useProject, useRun } from '../components/ui'

const KIND = { numeric: '연속형', categorical: '범주형', binary: '이진', datetime: '날짜', id: 'ID 의심', constant: '상수', text: '텍스트', high_cardinality: '고카디널리티' }
const ROLES = [
  { value: 'feature', label: '특징' },
  { value: 'id', label: '식별자' },
  { value: 'date', label: '날짜' },
  { value: 'group', label: '그룹' },
  { value: 'exclude', label: '제외' },
]

// ================================================================ 1단계
export function Step1() {
  const { pid, summary, refresh } = useProject()
  const r = useRun()
  const [files, setFiles] = useState([])
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState(null)
  const [drag, setDrag] = useState(false)
  const input = useRef()
  useEffect(() => {
    api.get('/data-files').then(setFiles).catch(() => {})
  }, [])
  const upload = async (list) => {
    if (!list?.length) return
    setBusy(true)
    setMsg(null)
    try {
      await api.upload(`/projects/${pid}/upload`, list)
      await refresh()
      r.run('s1_profile')
    } catch (e) {
      setMsg(e.message)
    }
    setBusy(false)
  }
  const addServer = async (name) => {
    setBusy(true)
    setMsg(null)
    try {
      await api.post(`/projects/${pid}/sources/server`, { name })
      await refresh()
      r.run('s1_profile')
    } catch (e) {
      setMsg(e.message)
    }
    setBusy(false)
  }
  const reload = async (s, patch) => {
    setBusy(true)
    try {
      await api.post(`/projects/${pid}/sources/${s.id}/reload`, patch)
      await refresh()
      r.run('s1_profile')
    } catch (e) {
      setMsg(e.message)
    }
    setBusy(false)
  }
  const remove = async (s) => {
    if (!window.confirm(`${s.name}을(를) 프로젝트에서 뺄까요?`)) return
    await api.del(`/projects/${pid}/sources/${s.id}`)
    refresh()
  }
  return (
    <>
      <Section title="파일 불러오기" desc="여러 파일을 한 번에 올릴 수 있습니다. 합계 1GB까지 지원합니다.">
        <div
          className={'drop' + (drag ? ' over' : '')}
          onDragOver={(e) => { e.preventDefault(); setDrag(true) }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => { e.preventDefault(); setDrag(false); upload([...e.dataTransfer.files]) }}
          onClick={() => input.current.click()}
        >
          <input ref={input} type="file" multiple accept=".csv,.xlsx,.xlsm,.xls,.parquet,.tsv,.txt" hidden onChange={(e) => upload([...e.target.files])} />
          <b>{busy ? '불러오는 중…' : '여기에 파일을 끌어다 놓거나 클릭해서 선택'}</b>
          <span className="muted">CSV(인코딩 자동 판별) · XLSX(시트 선택) · Parquet</span>
        </div>
        {files.length > 0 && (
          <div className="server-files">
            <span className="muted small">서버 data 폴더:</span>
            {files.map((f) => (
              <button key={f.name} className="ghost small" disabled={busy} onClick={() => addServer(f.name)}>
                + {f.name} <em className="muted">({(f.size / 1e6).toFixed(1)}MB)</em>
              </button>
            ))}
          </div>
        )}
        {msg && <div className="note warn">{msg}</div>}
      </Section>
      {summary.sources.length > 0 && (
        <Section title="불러온 파일" actions={<RunButton r={r} name="s1_profile">프로파일 다시 실행</RunButton>}>
          <div className="table-wrap">
            <table>
              <thead><tr><th>파일</th><th className="num">행</th><th className="num">열</th><th>형식</th><th>인코딩</th><th>시트</th><th>해시</th><th /></tr></thead>
              <tbody>
                {summary.sources.map((s) => (
                  <tr key={s.id}>
                    <td><b>{s.name}</b> <span className="muted small">{s.origin === 'server' ? '서버' : '업로드'}</span></td>
                    <td className="num">{s.rows?.toLocaleString()}</td>
                    <td className="num">{s.cols}</td>
                    <td>{s.format}</td>
                    <td>
                      {s.format === 'csv' ? (
                        <select value={s.encoding || ''} onChange={(e) => reload(s, { encoding: e.target.value })}>
                          {['utf-8-sig', 'utf-8', 'cp949', 'euc-kr', 'latin1'].map((x) => <option key={x}>{x}</option>)}
                        </select>
                      ) : '-'}
                    </td>
                    <td>
                      {s.sheets?.length > 1 ? (
                        <select value={s.sheet || ''} onChange={(e) => reload(s, { sheet: e.target.value })}>
                          {s.sheets.map((x) => <option key={x}>{x}</option>)}
                        </select>
                      ) : s.sheet ?? '-'}
                    </td>
                    <td className="muted small">{s.hash}</td>
                    <td><button className="ghost small danger" onClick={() => remove(s)}>삭제</button></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <RunStatus r={r} />
        </Section>
      )}
      {summary.columns.length > 0 && <Roles />}
      <Result rkey="s1" title="데이터 확인과 자동 프로파일" />
    </>
  )
}

function Roles() {
  const { pid, summary, refresh } = useProject()
  const [roles, setRoles] = useState({})
  const [saved, setSaved] = useState(false)
  useEffect(() => setRoles(Object.fromEntries(summary.columns.map((c) => [c.name, c.role]))), [summary.columns])
  const dirty = summary.columns.some((c) => roles[c.name] && roles[c.name] !== c.role)
  const save = async () => {
    await api.put(`/projects/${pid}/roles`, { roles })
    await refresh()
    setSaved(true)
    setTimeout(() => setSaved(false), 2000)
  }
  return (
    <Section title="열 역할 지정" desc="자동 추론한 역할입니다. 모델 특징에서 뺄 열(식별자·사후 측정값 등)은 '제외'로 바꾸세요. 종속변수는 4단계에서 정합니다."
      actions={<button className="primary" disabled={!dirty} onClick={save}>{saved ? '저장됨 ✓' : '역할 저장'}</button>}>
      <div className="table-wrap" style={{ maxHeight: 360 }}>
        <table>
          <thead><tr><th>열</th><th>추론 타입</th><th className="num">결측</th><th className="num">고유값</th><th>역할</th></tr></thead>
          <tbody>
            {summary.columns.map((c) => (
              <tr key={c.name}>
                <td><b>{c.name}</b></td>
                <td><span className={`tag k-${c.kind}`}>{KIND[c.kind] || c.kind}</span></td>
                <td className="num">{c.missing.toLocaleString()}</td>
                <td className="num">{c.nunique.toLocaleString()}</td>
                <td>
                  {c.role === 'target' ? <span className="tag target">타깃</span> : (
                    <select value={roles[c.name] || c.role} onChange={(e) => setRoles({ ...roles, [c.name]: e.target.value })}>
                      {ROLES.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
                    </select>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Section>
  )
}

// ================================================================ 2단계
export function Step2() {
  const { summary } = useProject()
  const r = useRun()
  const srcs = summary.sources
  const [sel, setSel] = useState(srcs.map((s) => s.id))
  const [mode, setMode] = useState(summary.merge?.mode && summary.merge.mode !== 'single' ? summary.merge.mode : srcs.length > 1 ? 'cols' : 'single')
  const [how, setHow] = useState(summary.merge?.how || 'inner')
  const [keys, setKeys] = useState(summary.merge?.keys || [])
  const { pid } = useProject()
  const chosen = sel.map((id) => srcs.find((s) => s.id === id)).filter(Boolean)
  return (
    <>
      <Need when={!srcs.length}>먼저 1단계에서 파일을 불러오세요.</Need>
      <Section title="결합 설정" desc={srcs.length < 2 ? '파일이 하나뿐이면 결합 없이 그대로 사용합니다.' : '목록의 첫 파일이 기준(왼쪽) 데이터입니다.'}>
        <Grid cols={2}>
          <Field label="결합할 데이터 (선택 순서 = 결합 순서)">
            <MultiSelect value={sel} onChange={setSel} options={srcs.map((s) => s.id)} labels={Object.fromEntries(srcs.map((s) => [s.id, `${s.name} · ${s.rows?.toLocaleString()}행`]))} height={140} />
          </Field>
          <div>
            <Field label="결합 방향">
              <Radio value={mode} onChange={setMode} options={[{ value: 'single', label: '결합 안 함(첫 파일)' }, { value: 'rows', label: '행 방향 (아래로 이어 붙이기)' }, { value: 'cols', label: '열 방향 (키로 옆에 붙이기)' }]} />
            </Field>
            {mode === 'cols' && (
              <>
                <Field label="결합 방법">
                  <Radio value={how} onChange={setHow} options={[{ value: 'inner', label: 'inner' }, { value: 'left', label: 'left' }, { value: 'right', label: 'right' }, { value: 'outer', label: 'outer' }]} />
                </Field>
                {chosen.slice(1).map((s, i) => (
                  <Field key={s.id} label={`${chosen[0]?.name} ⟷ ${s.name} 결합 키`} hint="비워 두면 공통 열 전체를 키로 사용합니다(기준 스크립트 방식).">
                    <KeyPicker pid={pid} a={chosen[0]} b={s} value={keys[i] || []} onChange={(v) => { const k = [...keys]; k[i] = v; setKeys(k) }} />
                  </Field>
                ))}
              </>
            )}
          </div>
        </Grid>
        <div className="actions">
          {mode === 'cols' && chosen.length > 1 && <RunButton r={r} name="s2_keys" params={{ sources: sel }} primary={false}>결합 키 후보 진단</RunButton>}
          <RunButton r={r} name="s2_merge" params={() => ({ mode, sources: sel, how, keys })} disabled={!sel.length}>데이터 결합 실행</RunButton>
        </div>
        <RunStatus r={r} />
      </Section>
      <Result rkey="s2_keys" title="결합 키 후보" />
      <Result rkey="s2" title="결합 결과와 품질 진단" empty={srcs.length < 2 ? '파일이 하나라 결합이 필요하지 않습니다. 그대로 다음 단계로 진행하세요.' : null} />
    </>
  )
}

function KeyPicker({ pid, a, b, value, onChange }) {
  const [common, setCommon] = useState(null)
  useEffect(() => {
    api.get(`/projects/${pid}/results/s1`).then((res) => {
      // 프로파일 결과에서 원본별 열 목록 추출
      const tabs = res?.blocks?.find((x) => x.type === 'tabs')?.tabs || [{ label: a.name, blocks: res?.blocks || [] }]
      const colsOf = (name) => {
        const t = tabs.find((t) => t.label === name)
        const prof = t?.blocks?.find((x) => x.type === 'table' && x.title?.startsWith('변수 정보'))
        return prof ? prof.rows.map((r) => r[0]) : []
      }
      const ca = colsOf(a.name), cb = new Set(colsOf(b.name))
      setCommon(ca.filter((c) => cb.has(c)))
    })
  }, [pid, a?.name, b?.name])
  if (!common) return <span className="muted small">1단계 프로파일을 먼저 실행하세요.</span>
  return <MultiSelect value={value} onChange={onChange} options={common} height={110} />
}

// ================================================================ 3단계
const OPS = [
  { value: 'bin', label: '1) binning 구간화' },
  { value: 'dummy', label: '2) dummy 더미변수' },
  { value: 'tukey', label: '3) Tukey 거듭제곱' },
  { value: 'arith', label: '4) 사칙연산' },
  { value: 'transform', label: '5) transform 변환' },
  { value: 'date', label: '날짜 분해' },
  { value: 'groupagg', label: '그룹 집계' },
  { value: 'condition', label: '조건식 (if-then)' },
  { value: 'lump', label: '희소 범주 병합' },
  { value: 'expr', label: '수식 직접 입력' },
]

export function Step3() {
  const { summary, setStep } = useProject()
  const r = useRun()
  const c = useColumns()
  const [op, setOp] = useState('bin')
  const [f, setF] = useState({ n_bins: 3, method: 'equal', power: 1, operator: '/', kind: 'log1p', parts: ['year', 'month', 'weekday'], agg: 'mean', top_n: 10, condop: '>', true_val: 1, false_val: 0 })
  const set = (k) => (v) => setF((o) => ({ ...o, [k]: v }))
  const num = c.numeric, all = c.all
  const params = () => {
    const base = { op, new: f.new }
    switch (op) {
      case 'bin': return { op: { ...base, source: f.source, n_bins: f.n_bins, method: f.method } }
      case 'dummy': return { op: { op, source: f.source, new: f.new || f.source, drop_first: !!f.drop_first } }
      case 'tukey': return { op: { ...base, source: f.source, power: f.power } }
      case 'arith': return { op: { ...base, left: f.left, right: f.right, operator: f.operator } }
      case 'transform': return { op: { ...base, source: f.source, kind: f.kind } }
      case 'date': return { op: { op, source: f.source, parts: f.parts } }
      case 'groupagg': return { op: { ...base, group: f.group, source: f.source, agg: f.agg, deviation: !!f.deviation } }
      case 'condition': return { op: { ...base, source: f.source, operator: f.condop, value: f.value, true_val: f.true_val, false_val: f.false_val } }
      case 'lump': return { op: { ...base, source: f.source, top_n: f.top_n } }
      case 'expr': return { op: { ...base, expr: f.expr } }
      default: return {}
    }
  }
  const remove = (i) => r.run('s3_remove', { index: i })
  return (
    <>
      <Need when={!summary.columns.length}>먼저 1단계에서 데이터를 불러오세요.</Need>
      <Section title="파생변수 만들기">
        <Grid cols={3}>
          <Field label="생성 방식"><Select value={op} onChange={setOp} options={OPS} /></Field>
          {op !== 'date' && <Field label={op === 'dummy' ? '더미 접두사' : '파생변수명'}><Text value={f.new} onChange={set('new')} placeholder={op === 'dummy' ? '(원래 열 이름)' : 'new_feature'} /></Field>}
          {['bin', 'tukey', 'transform'].includes(op) && <Field label="기준 연속형 변수"><Select value={f.source} onChange={set('source')} options={num} placeholder="선택" /></Field>}
          {['dummy', 'lump'].includes(op) && <Field label="기준 범주형 변수"><Select value={f.source} onChange={set('source')} options={c.categorical} placeholder="선택" /></Field>}
          {op === 'date' && <Field label="날짜 변수"><Select value={f.source} onChange={set('source')} options={c.cols.filter((x) => x.kind === 'datetime' || x.role === 'date').map((x) => x.name).concat(all)} placeholder="선택" /></Field>}
          {op === 'bin' && <>
            <Field label="구간 분할 방식"><Radio value={f.method} onChange={set('method')} options={[{ value: 'equal', label: '등간격' }, { value: 'quantile', label: '분위수' }]} /></Field>
            <Field label="구간 수"><Num value={f.n_bins} onChange={set('n_bins')} min={2} max={20} step={1} /></Field>
          </>}
          {op === 'dummy' && <Field label=" "><Check checked={f.drop_first} onChange={set('drop_first')} label="첫 번째 범주 제외 (다중공선성 방지)" /></Field>}
          {op === 'tukey' && <Field label={`Tukey's ladder of power: ${f.power}`} hint="-3 ~ 3, 0은 log(x+1)"><input type="range" min={-3} max={3} step={0.5} value={f.power} onChange={(e) => set('power')(Number(e.target.value))} /></Field>}
          {op === 'arith' && <>
            <Field label="왼쪽 변수"><Select value={f.left} onChange={set('left')} options={num} placeholder="선택" /></Field>
            <Field label="연산"><Radio value={f.operator} onChange={set('operator')} options={['+', '-', '*', '/']} /></Field>
            <Field label="오른쪽 변수"><Select value={f.right} onChange={set('right')} options={num} placeholder="선택" /></Field>
          </>}
          {op === 'transform' && <Field label="변환 방식"><Select value={f.kind} onChange={set('kind')} options={[
            { value: 'log1p', label: 'log1p (log(x+1))' }, { value: 'log', label: 'log' }, { value: 'sqrt', label: 'sqrt' }, { value: 'square', label: '제곱' },
            { value: 'reciprocal', label: '역수 1/x' }, { value: 'zscore', label: 'z-score 표준화' }, { value: 'minmax', label: '0~1 정규화' },
            { value: 'boxcox', label: 'Box-Cox (자동 λ)' }, { value: 'yeojohnson', label: 'Yeo-Johnson (자동 λ)' }]} /></Field>}
          {op === 'date' && <Field label="추출 요소"><MultiSelect value={f.parts} onChange={set('parts')} options={['year', 'quarter', 'month', 'day', 'weekday', 'hour', 'elapsed']} labels={{ year: '연', quarter: '분기', month: '월', day: '일', weekday: '요일', hour: '시', elapsed: '경과일' }} height={120} /></Field>}
          {op === 'groupagg' && <>
            <Field label="그룹 변수"><Select value={f.group} onChange={set('group')} options={c.categorical} placeholder="선택" /></Field>
            <Field label="집계할 변수"><Select value={f.source} onChange={set('source')} options={num} placeholder="선택" /></Field>
            <Field label="집계 함수"><Select value={f.agg} onChange={set('agg')} options={[{ value: 'mean', label: '평균' }, { value: 'median', label: '중앙값' }, { value: 'std', label: '표준편차' }, { value: 'min', label: '최소' }, { value: 'max', label: '최대' }, { value: 'count', label: '건수' }]} /></Field>
            <Field label=" "><Check checked={f.deviation} onChange={set('deviation')} label="그룹 평균과의 편차로 만들기" /></Field>
          </>}
          {op === 'condition' && <>
            <Field label="조건 변수"><Select value={f.source} onChange={set('source')} options={all} placeholder="선택" /></Field>
            <Field label="연산자"><Select value={f.condop} onChange={set('condop')} options={['>', '>=', '<', '<=', '==', '!=', 'in']} /></Field>
            <Field label="비교 값" hint="in은 쉼표로 구분"><Text value={f.value} onChange={set('value')} /></Field>
            <Field label="참일 때 값"><Text value={f.true_val} onChange={set('true_val')} /></Field>
            <Field label="거짓일 때 값"><Text value={f.false_val} onChange={set('false_val')} /></Field>
          </>}
          {op === 'lump' && <Field label="남길 상위 범주 수"><Num value={f.top_n} onChange={set('top_n')} min={1} step={1} /></Field>}
          {op === 'expr' && <Field label="수식" hint="예: (Ni_pct + Co_pct) / Mn_pct  — 열 이름에 공백이 있으면 `백틱`으로 감싸세요" wide><Text value={f.expr} onChange={set('expr')} /></Field>}
        </Grid>
        <div className="actions">
          {op === 'tukey' && f.source && <RunButton r={r} name="s3_tukey" params={{ column: f.source }} primary={false}>거듭제곱별 왜도 비교</RunButton>}
          <RunButton r={r} name="s3_derive" params={params}>파생변수 생성</RunButton>
        </div>
        <RunStatus r={r} />
      </Section>
      <Section title="레시피" desc="결합 데이터에 순서대로 적용되는 처리 목록입니다. 신규 데이터 예측(16단계)과 직접 입력 예측(13단계)에 자동으로 재적용됩니다.">
        {summary.recipe.length ? (
          <ol className="recipe">
            {summary.recipe.map((x) => (
              <li key={x.index}><span>{x.text}</span><button className="ghost small danger" onClick={() => remove(x.index)} disabled={r.running}>삭제</button></li>
            ))}
          </ol>
        ) : <p className="muted">아직 없습니다.</p>}
      </Section>
      <Result rkey="s3_tukey" title="Tukey 거듭제곱 비교" />
      <Result rkey="s3" title="파생변수 생성 결과" />
    </>
  )
}

// ================================================================ 4단계
export function Step4() {
  const { summary } = useProject()
  const r = useRun()
  const c = useColumns()
  const [target, setTarget] = useState(summary.target || '')
  const [ttype, setTtype] = useState('auto')
  const [pos, setPos] = useState(summary.classes?.[1] || '')
  const col = c.cols.find((x) => x.name === target)
  const [classes, setClasses] = useState([])
  const { pid } = useProject()
  useEffect(() => {
    if (!target) return
    // 범주 후보는 s1 프로파일의 최빈값 대신 직접 계산할 수 없으므로 nunique로 판단
    setClasses(summary.target === target && summary.classes ? summary.classes : [])
  }, [target, summary.target, summary.classes])
  const tt = ttype === 'auto' ? undefined : ttype
  return (
    <>
      <Section title="종속변수 선택">
        <Grid cols={3}>
          <Field label="종속변수"><Select value={target} onChange={setTarget} options={c.all} placeholder="선택" /></Field>
          <Field label="종속변수 타입" hint={col ? `고유값 ${col.nunique}개 · ${col.numeric ? '숫자' : '문자'}` : ''}>
            <Radio value={ttype} onChange={setTtype} options={[{ value: 'auto', label: '자동' }, { value: 'regression', label: '연속형' }, { value: 'classification', label: '범주형' }]} />
          </Field>
          {(summary.target_type === 'binary' && summary.target === target) && (
            <Field label="양성(관심) 클래스" hint="불량·이탈처럼 찾고 싶은 클래스">
              <Select value={pos} onChange={setPos} options={classes} />
            </Field>
          )}
        </Grid>
        <div className="actions">
          <RunButton r={r} name="s4_target" params={() => ({ target, ttype: tt, positive: pos || undefined })} disabled={!target}>종속변수 설정·분포 확인</RunButton>
        </div>
        <RunStatus r={r} />
      </Section>
      <Result rkey="s4" title="종속변수 분포" />
    </>
  )
}
