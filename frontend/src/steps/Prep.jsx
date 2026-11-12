import { useState } from 'react'
import { Check, Field, Grid, MultiSelect, Need, Num, Radio, Result, RunButton, RunStatus, Section, Select, useColumns, useProject, useRun } from '../components/ui'

// ================================================================ 5단계
export function Step5() {
  const { summary } = useProject()
  const r = useRun()
  const c = useColumns()
  const [cols, setCols] = useState([])
  const [method, setMethod] = useState('median')
  const [value, setValue] = useState(0)
  const [k, setK] = useState(5)
  const learned = summary.learned_ops.filter((o) => o.op === 'impute' || o.op === 'indicator')
  return (
    <>
      <Need when={!summary.columns.length}>먼저 데이터를 불러오세요.</Need>
      <Section title="결측 진단" actions={<RunButton r={r} name="s5_overview">결측 진단 실행</RunButton>}>
        <p className="muted">결측 열 {c.missing.length}개: {c.missing.slice(0, 15).join(', ') || '없음'}{c.missing.length > 15 ? ' …' : ''}</p>
        <RunStatus r={r} />
      </Section>
      <Section title="결측 처리">
        <Grid cols={3}>
          <Field label="처리할 변수" hint="결측이 있는 열만 표시합니다.">
            <MultiSelect value={cols} onChange={setCols} options={c.missing.filter((x) => x !== summary.target)} height={160} />
          </Field>
          <Field label="처리 방법">
            <Select value={method} onChange={setMethod} options={[
              { value: 'drop', label: '결측 행 삭제' },
              { value: 'mean', label: '평균 대체' },
              { value: 'median', label: '중앙값 대체' },
              { value: 'mode', label: '최빈값 대체' },
              { value: 'constant', label: '상수 대체' },
              { value: 'knn', label: 'KNN 대체 (유사한 행의 값)' },
              { value: 'iterative', label: '반복 대체 (다른 변수로 회귀)' },
              { value: 'ffill', label: '직전 값 (시계열)' },
              { value: 'indicator', label: '결측 지시변수 추가 (0/1)' },
            ]} />
          </Field>
          <div>
            {method === 'constant' && <Field label="대체 값"><input value={value} onChange={(e) => setValue(e.target.value)} /></Field>}
            {method === 'knn' && <Field label="이웃 수 k"><Num value={k} onChange={setK} min={1} step={1} /></Field>}
            <p className="muted small">
              {method === 'drop' ? '행 삭제는 전체 데이터에 바로 적용됩니다.' : method === 'indicator' ? '결측 여부를 새 0/1 열로 남깁니다. 대체와 함께 쓰면 결측의 정보를 보존합니다.' :
                summary.integrity_mode === 'improved' ? '대체값은 모델링 때 학습 데이터에서만 계산해 평가 데이터에 적용합니다(누수 방지). 화면의 탐색 결과에는 전체 기준으로 미리 반영합니다.' : '기준 스크립트 방식: 전체 데이터에서 계산한 값으로 대체합니다.'}
            </p>
          </div>
        </Grid>
        <div className="actions">
          <RunButton r={r} name="s5_apply" params={() => ({ columns: cols, method, value, k })} disabled={!cols.length}>결측치 처리 실행</RunButton>
        </div>
        {learned.length > 0 && (
          <>
            <h4>적용된 결측 처리</h4>
            <ol className="recipe">
              {learned.map((o) => (
                <li key={o.index}><span>{o.text}</span><button className="ghost small danger" disabled={r.running} onClick={() => r.run('s5_undo', { index: o.index })}>취소</button></li>
              ))}
            </ol>
          </>
        )}
      </Section>
      <Result rkey="s5" title="결측 진단·처리 결과" />
    </>
  )
}

// ================================================================ 6단계
export function Step6() {
  const { summary } = useProject()
  const r = useRun()
  const c = useColumns()
  const nums = c.numeric.filter((x) => x !== summary.target || summary.target_type === 'regression')
  const [col, setCol] = useState(nums[0] || '')
  const [method, setMethod] = useState('iqr')
  const [factor, setFactor] = useState(1.5)
  const [lo, setLo] = useState(1)
  const [hi, setHi] = useState(99)
  const [lower, setLower] = useState('')
  const [upper, setUpper] = useState('')
  const [act, setAct] = useState('winsor')
  const [cols, setCols] = useState([])
  const [mcols, setMcols] = useState(c.numFeatures.slice(0, 6))
  const [mm, setMm] = useState('iforest')
  const [cont, setCont] = useState(0.02)
  const defaults = { iqr: 1.5, zscore: 3, mad: 3.5 }
  const p = () => ({ column: col, columns: cols.length ? cols : [col], method, factor: ['iqr', 'zscore', 'mad'].includes(method) ? factor : undefined,
    lower_pct: lo, upper_pct: hi, lower: lower === '' ? undefined : lower, upper: upper === '' ? undefined : upper, action: act })
  const clip = summary.learned_ops.filter((o) => o.op === 'clip')
  return (
    <>
      <Section title="단변량 이상치 탐지">
        <Grid cols={4}>
          <Field label="이상치 확인 변수"><Select value={col} onChange={setCol} options={nums} /></Field>
          <Field label="탐지 기준">
            <Select value={method} onChange={(m) => { setMethod(m); if (defaults[m]) setFactor(defaults[m]) }} options={[
              { value: 'iqr', label: 'IQR (Q1−k·IQR ~ Q3+k·IQR)' }, { value: 'zscore', label: 'Z-score (평균 ± k·σ)' },
              { value: 'mad', label: '수정 Z (중앙값 ± k·MAD)' }, { value: 'percentile', label: '백분위' }, { value: 'custom', label: '직접 범위 지정' }]} />
          </Field>
          {['iqr', 'zscore', 'mad'].includes(method) && <Field label="계수 k"><Num value={factor} onChange={setFactor} step={0.1} /></Field>}
          {method === 'percentile' && <Field label="하위·상위 백분위(%)"><div className="pair"><Num value={lo} onChange={setLo} /><Num value={hi} onChange={setHi} /></div></Field>}
          {method === 'custom' && <Field label="하한·상한"><div className="pair"><Num value={lower} onChange={setLower} placeholder="하한" /><Num value={upper} onChange={setUpper} placeholder="상한" /></div></Field>}
        </Grid>
        <div className="actions"><RunButton r={r} name="s6_detect" params={p} disabled={!col}>이상치 확인</RunButton></div>
        <h4>처리</h4>
        <Grid cols={2}>
          <Field label="처리 방법">
            <Radio value={act} onChange={setAct} options={[
              { value: 'none', label: '처리하지 않음' }, { value: 'drop', label: '이상치 행 제거' },
              { value: 'normal_minmax', label: '정상 범위 최소/최대로 대체' }, { value: 'winsor', label: '경계값으로 대체(윈저화)' }, { value: 'nan', label: '결측으로 바꾸기' }]} />
          </Field>
          <Field label="함께 처리할 변수 (비우면 위 변수만)"><MultiSelect value={cols} onChange={setCols} options={nums} height={120} /></Field>
        </Grid>
        <div className="actions"><RunButton r={r} name="s6_apply" params={p} disabled={!col}>이상치 처리 실행</RunButton></div>
        {clip.length > 0 && <ol className="recipe">{clip.map((o) => <li key={o.index}><span>{o.text}</span></li>)}</ol>}
        <RunStatus r={r} />
      </Section>
      <Result rkey="s6" title="이상치 확인·처리 결과" />
      <Section title="다변량 이상치 탐지 (심화)" desc="변수 하나씩 보면 정상이지만 조합이 특이한 행을 찾습니다.">
        <Grid cols={3}>
          <Field label="변수"><MultiSelect value={mcols} onChange={setMcols} options={nums} height={140} /></Field>
          <Field label="방법"><Radio value={mm} onChange={setMm} options={[{ value: 'iforest', label: 'Isolation Forest' }, { value: 'mahalanobis', label: '강건 마할라노비스' }]} /></Field>
          <Field label="예상 이상 비율"><Num value={cont} onChange={setCont} step={0.01} min={0.001} max={0.5} /></Field>
        </Grid>
        <div className="actions"><RunButton r={r} name="s6_multi" params={() => ({ columns: mcols, method: mm, contamination: cont })} disabled={mcols.length < 2}>다변량 탐지</RunButton></div>
      </Section>
      <Result rkey="s6_multi" title="다변량 이상치" />
    </>
  )
}

// ================================================================ 7단계
const EDA_TABS = [
  ['summary', '관련 변수 요약'], ['scatter', '산점도'], ['box', '박스플롯·그룹 비교'], ['corr', '상관분석'], ['partial', '편상관'],
  ['anova', 'ANOVA / 분류 타깃'], ['crosstab', '교차표'], ['vif', '다중공선성(VIF)'], ['mi', '상호정보량'], ['matrix', '산점도 행렬'], ['trend', '시간 추이'],
]

export function Step7() {
  const { summary } = useProject()
  const [tab, setTab] = useState('summary')
  return (
    <>
      <Need when={!summary.target}>4단계에서 종속변수를 먼저 정하면 타깃 기준 분석이 함께 표시됩니다.</Need>
      <div className="tabbar big">
        {EDA_TABS.map(([k, l]) => <button key={k} className={tab === k ? 'active' : ''} onClick={() => setTab(k)}>{l}</button>)}
      </div>
      <EdaTab key={tab} kind={tab} label={EDA_TABS.find((t) => t[0] === tab)[1]} />
    </>
  )
}

function EdaTab({ kind, label }) {
  const { summary } = useProject()
  const r = useRun()
  const c = useColumns()
  const tgt = summary.target
  const isClf = summary.target_type && summary.target_type !== 'regression'
  const [p, setP] = useState(() => ({
    x: c.numFeatures[0], y: tgt && !isClf ? tgt : c.numFeatures[1], trend: 'ols', max_rows: 2000, cat: c.categorical.find((x) => x !== tgt) || c.categorical[0],
    num: tgt && !isClf ? tgt : c.numFeatures[0], top_n: 20, columns: kind === 'corr' ? [...c.numFeatures.slice(0, 11), ...(tgt && !isClf ? [tgt] : [])] : kind === 'matrix' ? c.numFeatures.slice(0, 4) : [],
    method: 'pearson', controls: [], a: c.categorical[0], b: tgt || c.categorical[1], time: c.cols.find((x) => x.kind === 'datetime')?.name, values: [], freq: 'D', agg: 'mean', max_groups: 30, top: 10,
  }))
  const set = (k) => (v) => setP((o) => ({ ...o, [k]: v }))
  const run = () => r.run('s7', { kind: kind === 'anova' && isClf ? 'classtarget' : kind, ...p })
  const rkey = `s7_${kind === 'anova' && isClf ? 'classtarget' : kind}`
  const desc = {
    summary: '여러 지표로 타깃과 관련성이 큰 변수를 한눈에 정리합니다.',
    scatter: '두 연속형 변수의 관계와 추세선, Pearson·Spearman 상관을 봅니다.',
    box: '범주별 분포와 ANOVA·Kruskal·Tukey HSD 사후검정을 함께 봅니다.',
    corr: '상관 히트맵, 강한 변수 쌍, 타깃 기준 상관 순위를 계산합니다.',
    partial: '통제변수의 영향을 제거한 두 변수의 순수한 상관을 계산합니다.',
    anova: isClf ? '범주형 타깃: 카이제곱·Cramér\'s V와 클래스별 평균 차이를 봅니다.' : '범주형 변수별로 타깃 평균 차이를 ANOVA로 비교합니다.',
    crosstab: '두 범주형 변수의 빈도·비율과 독립성 검정을 봅니다.',
    vif: '특징 간 다중공선성을 분산팽창계수로 진단합니다.',
    mi: '선형·비선형 관계를 모두 잡는 상호정보량으로 변수를 순위화합니다.',
    matrix: '여러 연속형 변수의 쌍별 산점도를 한 번에 봅니다.',
    trend: '시간 열이 있으면 기간별 집계 추이와 이동평균을 봅니다.',
  }[kind]
  return (
    <>
      <Section title={label} desc={desc}>
        <Grid cols={3}>
          {kind === 'scatter' && <>
            <Field label="X축 연속형 변수"><Select value={p.x} onChange={set('x')} options={c.numeric} /></Field>
            <Field label="Y축 연속형 변수"><Select value={p.y} onChange={set('y')} options={c.numeric} /></Field>
            <Field label="색상 그룹 (선택)"><Select value={p.color} onChange={set('color')} options={c.categorical} allowEmpty /></Field>
            <Field label="추세선"><Radio value={p.trend} onChange={set('trend')} options={[{ value: 'none', label: '없음' }, { value: 'ols', label: '선형' }, { value: 'lowess', label: 'LOWESS' }]} /></Field>
            <Field label="그래프 최대 행 수" hint="계산은 전체, 그림은 표본"><Num value={p.max_rows} onChange={set('max_rows')} min={300} max={20000} step={100} /></Field>
            <Field label="hover 표시 변수"><MultiSelect value={p.hover || []} onChange={set('hover')} options={c.all} height={100} /></Field>
          </>}
          {kind === 'box' && <>
            <Field label="X축 범주형 변수"><Select value={p.cat} onChange={set('cat')} options={c.categorical} /></Field>
            <Field label="Y축 연속형 변수"><Select value={p.num} onChange={set('num')} options={c.numeric} /></Field>
            <Field label="상위 범주 수"><Num value={p.top_n} onChange={set('top_n')} min={3} max={100} step={1} /></Field>
          </>}
          {(kind === 'corr' || kind === 'vif' || kind === 'matrix' || kind === 'mi') && (
            <Field label={kind === 'mi' ? '분석할 특징 (비우면 전체)' : '변수 선택'}><MultiSelect value={p.columns} onChange={set('columns')} options={kind === 'mi' ? c.features : c.numeric} height={180} /></Field>
          )}
          {kind === 'corr' && <Field label="상관계수"><Radio value={p.method} onChange={set('method')} options={[{ value: 'pearson', label: 'Pearson' }, { value: 'spearman', label: 'Spearman' }, { value: 'kendall', label: 'Kendall' }]} /></Field>}
          {kind === 'matrix' && <Field label="색상 그룹"><Select value={p.color} onChange={set('color')} options={c.categorical} allowEmpty /></Field>}
          {kind === 'partial' && <>
            <Field label="변수 X"><Select value={p.x} onChange={set('x')} options={c.numeric} /></Field>
            <Field label="변수 Y"><Select value={p.y} onChange={set('y')} options={c.numeric} /></Field>
            <Field label="통제변수"><MultiSelect value={p.controls} onChange={set('controls')} options={c.numeric} height={140} /></Field>
          </>}
          {kind === 'anova' && !isClf && <>
            <Field label="범주형 변수 (비우면 상위 10개)"><MultiSelect value={p.columns} onChange={set('columns')} options={c.categorical.filter((x) => x !== tgt)} height={160} /></Field>
            <Field label="변수별 최대 범주 수"><Num value={p.max_groups} onChange={set('max_groups')} min={2} max={200} step={1} /></Field>
          </>}
          {kind === 'crosstab' && <>
            <Field label="행 변수"><Select value={p.a} onChange={set('a')} options={c.categorical} /></Field>
            <Field label="열 변수"><Select value={p.b} onChange={set('b')} options={c.categorical} /></Field>
          </>}
          {kind === 'trend' && <>
            <Field label="시간 열"><Select value={p.time} onChange={set('time')} options={c.all} placeholder="선택" /></Field>
            <Field label="값 변수"><MultiSelect value={p.values} onChange={set('values')} options={c.numeric} height={140} /></Field>
            <Field label="집계 단위"><Select value={p.freq} onChange={set('freq')} options={[{ value: 'none', label: '집계 안 함' }, { value: 'h', label: '시간' }, { value: 'D', label: '일' }, { value: 'W', label: '주' }, { value: 'MS', label: '월' }]} /></Field>
          </>}
          {kind === 'summary' && <Field label="상위 N개"><Num value={p.top} onChange={set('top')} min={3} max={50} step={1} /></Field>}
        </Grid>
        <div className="actions"><RunButton r={r} name="s7" params={() => ({ kind: kind === 'anova' && isClf ? 'classtarget' : kind, ...p })}>분석 실행</RunButton></div>
        <RunStatus r={r} />
      </Section>
      <Result rkey={rkey} title={label} />
    </>
  )
}
