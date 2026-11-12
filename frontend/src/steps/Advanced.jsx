import { useEffect, useMemo, useRef, useState } from 'react'
import { api, fmt } from '../api'
import { Check, Field, Grid, MultiSelect, Need, Num, Radio, Result, RunButton, RunStatus, Section, Select, useColumns, useProject, useRun } from '../components/ui'

// ---------------------------------------------------------------- 공유 입력 폼
function useSpec() {
  const { pid, summary } = useProject()
  const [spec, setSpec] = useState(null)
  const [err, setErr] = useState(null)
  useEffect(() => {
    if (!summary.best_model) return
    api.get(`/projects/${pid}/input-spec`).then(setSpec).catch((e) => setErr(e.message))
  }, [pid, summary.best_model, summary.split])
  return { spec, err }
}

function useRow(spec) {
  const { pid } = useProject()
  const key = `row_${pid}`
  const [row, setRow] = useState(() => {
    try {
      return JSON.parse(sessionStorage.getItem(key)) || {}
    } catch {
      return {}
    }
  })
  useEffect(() => {
    if (!spec) return
    setRow((r) => {
      const out = {}
      for (const s of spec.spec) out[s.name] = r[s.name] ?? (s.type === 'number' ? s.median : s.default)
      return out
    })
  }, [spec])
  useEffect(() => {
    try {
      sessionStorage.setItem(key, JSON.stringify(row))
    } catch {}
  }, [key, row])
  return [row, setRow]
}

function InputForm({ spec, row, setRow, compact }) {
  const { pid } = useProject()
  const [info, setInfo] = useState(null)
  const sample = async () => {
    const d = await api.get(`/projects/${pid}/sample-row`)
    const r = {}
    for (const s of spec.spec) r[s.name] = d.row[s.name] ?? (s.type === 'number' ? s.median : s.default)
    setRow(r)
    setInfo(`학습 데이터 ${d.index}번 행 · 실제 타깃 ${fmt(d.actual)}`)
  }
  const median = () => {
    const r = {}
    for (const s of spec.spec) r[s.name] = s.type === 'number' ? s.median : s.default
    setRow(r)
    setInfo('중앙값·최빈값으로 초기화')
  }
  return (
    <div>
      <div className="actions left">
        <button className="ghost small" onClick={sample}>학습 데이터에서 무작위 행 불러오기</button>
        <button className="ghost small" onClick={median}>중앙값으로 초기화</button>
        {info && <span className="muted small">{info}</span>}
      </div>
      <div className={'input-grid' + (compact ? ' compact' : '')}>
        {spec.spec.map((s) => (
          <label key={s.name} className="field">
            <span className="field-label">{s.name}</span>
            {s.type === 'number' ? (
              <>
                <input type="number" step="any" value={row[s.name] ?? ''} onChange={(e) => setRow({ ...row, [s.name]: e.target.value === '' ? '' : Number(e.target.value) })}
                  className={row[s.name] !== '' && (row[s.name] < s.min || row[s.name] > s.max) ? 'out' : ''} />
                <span className="field-hint">범위 {fmt(s.min)} ~ {fmt(s.max)} · 중앙값 {fmt(s.median)}</span>
              </>
            ) : (
              <select value={row[s.name] ?? ''} onChange={(e) => setRow({ ...row, [s.name]: e.target.value })}>
                {s.options.map((o) => <option key={o}>{o}</option>)}
              </select>
            )}
          </label>
        ))}
      </div>
    </div>
  )
}

// ================================================================ 13단계
export function Step13() {
  const { summary } = useProject()
  const r = useRun()
  const { spec, err } = useSpec()
  const [row, setRow] = useRow(spec)
  const [v, setV] = useState('')
  const [model, setModel] = useState('')
  const nums = spec?.spec.filter((s) => s.type === 'number').map((s) => s.name) || []
  return (
    <>
      <Need when={!summary.best_model}>10단계에서 모델을 먼저 학습하세요.</Need>
      {err && <div className="note warn">{err}</div>}
      {spec && (
        <Section title="독립변수 입력" desc="파생변수는 입력값에서 레시피로 자동 계산됩니다. 학습 범위를 벗어난 값은 빨간 테두리로 표시됩니다."
          actions={<Select value={model} onChange={setModel} options={(summary.models || []).filter((m) => m.key !== 'baseline').map((m) => ({ value: m.key, label: m.name }))} placeholder={`최적 모델 (${spec.model})`} />}>
          <InputForm spec={spec} row={row} setRow={setRow} />
          <div className="actions">
            <RunButton r={r} name="s13_sensitivity" params={() => ({ row, model: model || undefined })} primary={false}>전체 변수 민감도</RunButton>
            <RunButton r={r} name="s13_predict" params={() => ({ row, model: model || undefined })}>예측하기</RunButton>
          </div>
          <RunStatus r={r} />
        </Section>
      )}
      <Result rkey="s13" title="예측 결과" />
      {spec && (
        <Section title="What-if 분석 (심화)" desc="변수 하나를 범위 안에서 움직일 때 예측이 어떻게 바뀌는지 봅니다.">
          <Grid cols={3}>
            <Field label="움직일 변수"><Select value={v} onChange={setV} options={spec.spec.map((s) => s.name)} placeholder="선택" /></Field>
          </Grid>
          <div className="actions"><RunButton r={r} name="s13_whatif" params={() => ({ row, var: v, model: model || undefined })} disabled={!v}>What-if 실행</RunButton></div>
        </Section>
      )}
      <Result rkey="s13_whatif" title="What-if" />
      <Result rkey="s13_sens" title="민감도 (토네이도)" />
    </>
  )
}

// ================================================================ 14단계
const METHODS = [
  { value: 'backdoor.linear_regression', label: '선형회귀 (backdoor.linear_regression)' },
  { value: 'backdoor.propensity_score_matching', label: '성향점수 매칭 (이진 처치)' },
  { value: 'backdoor.propensity_score_stratification', label: '성향점수 층화 (이진 처치)' },
  { value: 'backdoor.propensity_score_weighting', label: '성향점수 가중 IPW (이진 처치)' },
]
const REFUTERS = [
  ['random_common_cause', '무작위 공통원인 추가'], ['placebo_treatment_refuter', '위약 처치'], ['data_subset_refuter', '데이터 부분집합'],
  ['bootstrap_refuter', '부트스트랩'], ['add_unobserved_common_cause', '미관측 교란 민감도'],
]

export function Step14() {
  const { summary } = useProject()
  const r = useRun()
  const c = useColumns()
  const feats = summary.split?.features || c.features
  const [t, setT] = useState(c.numFeatures[0] || feats[0] || '')
  const [ctrl, setCtrl] = useState(null)
  const [method, setMethod] = useState(METHODS[0].value)
  const [std, setStd] = useState(false)
  const [bin, setBin] = useState('')
  const [refs, setRefs] = useState(['random_common_cause', 'placebo_treatment_refuter', 'data_subset_refuter'])
  const [dml, setDml] = useState(true)
  const [oc, setOc] = useState('')
  const [edges, setEdges] = useState([])
  const [useDag, setUseDag] = useState(false)
  const [cands, setCands] = useState(c.numFeatures.filter((x) => feats.includes(x)))
  const [bstd, setBstd] = useState(true)
  const [seg, setSeg] = useState('')
  const controls = ctrl ?? feats.filter((x) => x !== t).slice(0, 10)
  const isClf = summary.target_type && summary.target_type !== 'regression'
  const dagNodes = [t, summary.target, ...controls].filter(Boolean)
  const addEdge = () => setEdges([...edges, [controls[0] || t, t]])
  return (
    <>
      <Need when={!summary.target}>4단계에서 종속변수를 먼저 정하세요.</Need>
      <Section title="DoWhy 인과분석: 처치변수별 인과효과 추정" desc="Model → Identify → Estimate → Refute 4단계를 실행합니다. 결측·이상치 처리를 반영한 원 단위 데이터로 분석합니다.">
        <Grid cols={3}>
          <div>
            <Field label="처치변수 (원인으로 볼 변수)"><Select value={t} onChange={(v) => { setT(v); setCtrl(null) }} options={feats} /></Field>
            <Field label="추정 방법"><Select value={method} onChange={setMethod} options={METHODS} /></Field>
            {(method.includes('propensity') || bin) && (
              <Field label="처치 이진화" hint="성향점수 방법은 0/1 처치만 가능합니다.">
                <Select value={bin} onChange={setBin} options={[{ value: 'median', label: '중앙값 초과 = 1' }]} allowEmpty />
              </Field>
            )}
            {isClf && <Field label="결과 클래스 (확률 효과)"><Select value={oc} onChange={setOc} options={summary.classes || []} placeholder={`기본: ${summary.classes?.at(-1)}`} /></Field>}
            <Check checked={std} onChange={setStd} label="표준화 단위로 추정 (1 표준편차당 효과)" />
            <Check checked={dml} onChange={setDml} label="이중 머신러닝(DML)로 교차 확인" />
          </div>
          <Field label={`통제(교란) 변수 ${controls.length}개`} hint="처치와 결과 모두에 영향을 주는 변수. 처치 이후에 정해지는 변수는 넣지 마세요.">
            <MultiSelect value={controls} onChange={setCtrl} options={feats.filter((x) => x !== t)} height={240} />
          </Field>
          <div>
            <Field label="반박 검정 (Refute)">
              {REFUTERS.map(([k, l]) => <Check key={k} checked={refs.includes(k)} onChange={(v) => setRefs(v ? [...refs, k] : refs.filter((x) => x !== k))} label={l} />)}
            </Field>
            <Check checked={useDag} onChange={setUseDag} label="인과 그래프(DAG) 직접 지정" />
          </div>
        </Grid>
        {useDag && (
          <div className="dag">
            <p className="muted small">화살표(원인 → 결과)를 지정하세요. 처치 → 결과 화살표는 자동으로 추가됩니다. 지정하지 않은 통제변수는 연결되지 않습니다.</p>
            {edges.map((e, i) => (
              <div key={i} className="edge">
                <select value={e[0]} onChange={(ev) => { const x = [...edges]; x[i] = [ev.target.value, e[1]]; setEdges(x) }}>{dagNodes.map((n) => <option key={n}>{n}</option>)}</select>
                <span>→</span>
                <select value={e[1]} onChange={(ev) => { const x = [...edges]; x[i] = [e[0], ev.target.value]; setEdges(x) }}>{dagNodes.map((n) => <option key={n}>{n}</option>)}</select>
                <button className="ghost small danger" onClick={() => setEdges(edges.filter((_, j) => j !== i))}>삭제</button>
              </div>
            ))}
            <button className="ghost small" onClick={addEdge}>+ 화살표 추가</button>
            <button className="ghost small" onClick={() => setEdges(controls.flatMap((x) => [[x, t], [x, summary.target]]))}>기본 구조(모든 통제변수 → 처치·결과)로 채우기</button>
          </div>
        )}
        <div className="actions">
          <RunButton r={r} name="s14_causal" params={() => ({ treatment: t, controls, method, standardize: std, binarize: bin || undefined, refuters: refs, dml, outcome_class: oc || undefined, edges: useDag && edges.length ? edges : undefined })} disabled={!t || !summary.target}>선택 변수로 DoWhy 실행</RunButton>
        </div>
        <RunStatus r={r} />
      </Section>
      <Result rkey="s14" title="DoWhy 4단계 결과" />
      <Section title="변수 일괄 인과효과 (심화)" desc="모든 후보 변수에 대해 나머지를 통제한 효과를 계산하고, 11~12단계 예측 중요도와 비교합니다.">
        <Grid cols={3}>
          <Field label="후보 변수"><MultiSelect value={cands} onChange={setCands} options={feats} height={180} /></Field>
          <Field label="단위"><Radio value={bstd ? 'std' : 'raw'} onChange={(v) => setBstd(v === 'std')} options={[{ value: 'std', label: '표준화(크기 비교용)' }, { value: 'raw', label: '원 단위' }]} /></Field>
        </Grid>
        <div className="actions"><RunButton r={r} name="s14_batch" params={() => ({ candidates: cands, standardize: bstd, outcome_class: oc || undefined })} disabled={cands.length < 2}>일괄 인과효과 계산</RunButton></div>
      </Section>
      <Result rkey="s14_batch" title="변수 일괄 인과효과" />
      <Section title="이질적 효과 (심화)" desc="조건(구간·그룹)에 따라 처치 효과가 달라지는지 봅니다.">
        <Grid cols={3}>
          <Field label="구간을 나눌 변수"><Select value={seg} onChange={setSeg} options={c.all.filter((x) => x !== summary.target && x !== t)} placeholder="선택" /></Field>
        </Grid>
        <div className="actions"><RunButton r={r} name="s14_hetero" params={() => ({ treatment: t, controls, segment: seg, standardize: std, outcome_class: oc || undefined })} disabled={!seg}>구간별 효과 계산</RunButton></div>
      </Section>
      <Result rkey="s14_hetero" title="이질적 효과" />
    </>
  )
}

// ================================================================ 15단계
export function Step15() {
  const { summary } = useProject()
  const r = useRun()
  const { spec, err } = useSpec()
  const [row, setRow] = useRow(spec)
  const isReg = summary.target_type === 'regression'
  const [mut, setMut] = useState([])
  const [dir, setDir] = useState('이하')
  const [tv, setTv] = useState('')
  const [tc, setTc] = useState(summary.classes?.at(-1) || '')
  const [mp, setMp] = useState(0.5)
  const [method, setMethod] = useState('random')
  const [n, setN] = useState(3000)
  const [k, setK] = useState(5)
  const [seed, setSeed] = useState(42)
  const [mc, setMc] = useState(4)
  const [cons, setCons] = useState({})
  const [scope, setScope] = useState('test')
  const [nRows, setNRows] = useState(100)
  useEffect(() => {
    if (spec && !mut.length) setMut(spec.spec.filter((s) => s.type === 'number').slice(0, 8).map((s) => s.name))
  }, [spec])
  const rule = () => (isReg ? { direction: dir, target_value: tv } : { target_class: tc, min_prob: mp })
  const common = () => ({ row, mutable: mut, ...rule(), constraints: cons })
  const numMut = spec ? mut.filter((m) => spec.spec.find((s) => s.name === m)?.type === 'number') : []
  const setC = (m, key, v) => setCons({ ...cons, [m]: { ...(cons[m] || {}), [key]: v } })
  return (
    <>
      <Need when={!summary.best_model}>10단계에서 모델을 먼저 학습하세요.</Need>
      {err && <div className="note warn">{err}</div>}
      {spec && (
        <>
          <Section title="현재 조건 입력" desc="13단계와 같은 입력값을 공유합니다.">
            <InputForm spec={spec} row={row} setRow={setRow} compact />
          </Section>
          <Section title="Counterfactual 목표 설정">
            <Grid cols={3}>
              <Field label="변경 가능한 독립변수" hint="현장에서 실제로 조정할 수 있는 변수만 고르세요. 원료 특성·환경·ID는 제외합니다.">
                <MultiSelect value={mut} onChange={setMut} options={spec.spec.map((s) => s.name)} height={220} />
              </Field>
              <div>
                {isReg ? (
                  <>
                    <Field label="목표 방향"><Radio value={dir} onChange={setDir} options={['이하', '이상']} /></Field>
                    <Field label="목표 예측값"><Num value={tv} onChange={setTv} /></Field>
                  </>
                ) : (
                  <>
                    <Field label="목표 클래스"><Select value={tc} onChange={setTc} options={summary.classes || []} /></Field>
                    <Field label={`목표 클래스 최소 확률: ${mp}`}><input type="range" min={0} max={1} step={0.05} value={mp} onChange={(e) => setMp(Number(e.target.value))} /></Field>
                  </>
                )}
                <Field label="탐색 방법"><Radio value={method} onChange={setMethod} options={[{ value: 'random', label: '무작위 탐색 (기준)' }, { value: 'genetic', label: '유전 알고리즘' }]} /></Field>
              </div>
              <div>
                <Field label="탐색 후보 수"><Num value={n} onChange={setN} min={200} max={50000} step={200} /></Field>
                <Field label="추천 대안 수"><Num value={k} onChange={setK} min={1} max={20} step={1} /></Field>
                <Field label="최대 변경 변수 수"><Num value={mc} onChange={setMc} min={1} max={20} step={1} /></Field>
                <Field label="Random seed"><Num value={seed} onChange={setSeed} step={1} /></Field>
              </div>
            </Grid>
            {numMut.length > 0 && (
              <details className="constraints">
                <summary>변수별 제약 (허용 범위·방향·변경 비용) — 비우면 학습 데이터 1~99% 범위</summary>
                <div className="table-wrap">
                  <table>
                    <thead><tr><th>변수</th><th>최소</th><th>최대</th><th>방향</th><th>변경 비용(가중치)</th></tr></thead>
                    <tbody>
                      {numMut.map((m) => {
                        const s = spec.spec.find((x) => x.name === m)
                        return (
                          <tr key={m}>
                            <td>{m}</td>
                            <td><input type="number" step="any" placeholder={fmt(s.q01)} value={cons[m]?.min ?? ''} onChange={(e) => setC(m, 'min', e.target.value)} /></td>
                            <td><input type="number" step="any" placeholder={fmt(s.q99)} value={cons[m]?.max ?? ''} onChange={(e) => setC(m, 'max', e.target.value)} /></td>
                            <td><select value={cons[m]?.direction || 'both'} onChange={(e) => setC(m, 'direction', e.target.value)}><option value="both">양방향</option><option value="increase">증가만</option><option value="decrease">감소만</option></select></td>
                            <td><input type="number" step="0.1" placeholder="1" value={cons[m]?.cost ?? ''} onChange={(e) => setC(m, 'cost', e.target.value)} /></td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              </details>
            )}
            <div className="actions">
              <RunButton r={r} name="s15_cf" params={() => ({ ...common(), method, n_samples: n, top_k: k, seed, max_change: mc })} disabled={!mut.length || (isReg && tv === '')}>Counterfactual XAI 실행</RunButton>
            </div>
            <RunStatus r={r} />
          </Section>
          <Result rkey="s15" title="추천 변경안" />
          <Section title="일괄 Counterfactual · 최적 조건 탐색 (심화)" desc="위의 목표·변경 변수·제약을 그대로 사용합니다.">
            <Grid cols={3}>
              <Field label="대상 데이터"><Radio value={scope} onChange={setScope} options={[{ value: 'test', label: '평가 데이터' }, { value: 'all', label: '전체 데이터' }]} /></Field>
              <Field label="분석할 최대 행 수"><Num value={nRows} onChange={setNRows} min={5} max={2000} step={5} /></Field>
            </Grid>
            <div className="actions">
              <RunButton r={r} name="s15_batch" params={() => ({ ...common(), scope, n_rows: nRows, n_samples: 400, max_change: mc })} primary={false} disabled={!mut.length || (isReg && tv === '')}>목표 미달 행 일괄 개선안</RunButton>
              <RunButton r={r} name="s15_optimize" params={() => ({ ...common(), n_samples: 300 })} disabled={!mut.length || (isReg && tv === '')}>목표 달성 최적 조건 탐색</RunButton>
            </div>
          </Section>
          <Result rkey="s15_batch" title="일괄 Counterfactual" />
          <Result rkey="s15_opt" title="최적 조건 탐색" />
        </>
      )}
    </>
  )
}

// ================================================================ 16단계
export function Step16() {
  const { summary, pid, refresh } = useProject()
  const r = useRun()
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState(null)
  const [sel, setSel] = useState('')
  const input = useRef()
  const nds = summary.newdata || []
  const upload = async (f) => {
    if (!f) return
    setBusy(true)
    setMsg(null)
    try {
      const res = await api.upload(`/projects/${pid}/newdata`, [f], 'file')
      await refresh()
      setSel(res.added.id)
      r.run('s16_predict', { file: res.added.id })
    } catch (e) {
      setMsg(e.message)
    }
    setBusy(false)
  }
  return (
    <>
      <Need when={!summary.best_model}>10단계에서 모델을 먼저 학습하세요.</Need>
      <Section title="신규 데이터 업로드 (CSV/XLSX)" desc="학습에 쓴 원본과 같은 형식의 파일을 올리면 3단계 파생변수와 전처리를 자동으로 적용해 예측합니다. 종속변수 열이 있으면 성능도 평가합니다.">
        <div className="drop" onClick={() => input.current.click()}>
          <input ref={input} type="file" hidden accept=".csv,.xlsx,.xls,.parquet" onChange={(e) => upload(e.target.files[0])} />
          <b>{busy ? '업로드 중…' : '클릭해서 신규 데이터 선택'}</b>
        </div>
        {msg && <div className="note warn">{msg}</div>}
        {nds.length > 0 && (
          <div className="actions left">
            <Select value={sel} onChange={setSel} options={nds.map((n) => ({ value: n.id, label: `${n.name} · ${n.rows.toLocaleString()}행` }))} placeholder="업로드한 파일 선택" />
            <RunButton r={r} name="s16_predict" params={() => ({ file: sel })} disabled={!sel}>예측 실행</RunButton>
          </div>
        )}
        <RunStatus r={r} />
      </Section>
      <Result rkey="s16" title="신규 데이터 예측 결과" />
    </>
  )
}
