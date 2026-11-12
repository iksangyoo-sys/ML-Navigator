import { useEffect, useState } from 'react'
import { api, fmt } from '../api'
import { Check, Field, Grid, MultiSelect, Need, Num, Radio, Result, RunButton, RunStatus, Section, Select, useColumns, useProject, useRun } from '../components/ui'

// ================================================================ 8단계
export function Step8() {
  const { summary } = useProject()
  const r = useRun()
  const c = useColumns()
  const sp = summary.split
  const isClf = summary.target_type && summary.target_type !== 'regression'
  const [feats, setFeats] = useState(sp?.features || c.features)
  const [method, setMethod] = useState(sp?.method || (isClf ? 'stratified' : 'random'))
  const [size, setSize] = useState(sp?.test_size || 0.3)
  const [seed, setSeed] = useState(sp?.seed ?? 42)
  const [group, setGroup] = useState(sp?.group_col || '')
  const [time, setTime] = useState(sp?.time_col || '')
  const [dropna, setDropna] = useState(false)
  const pool = c.all.filter((x) => x !== summary.target)
  return (
    <>
      <Need when={!summary.target}>4단계에서 종속변수를 먼저 정하세요.</Need>
      <Section title="학습/평가 데이터 분할">
        <Grid cols={3}>
          <Field label={`독립변수 (${feats.length}개)`} hint="ID·사후 측정값은 빼세요. 1단계 역할이 '특징'인 열이 기본 선택됩니다.">
            <MultiSelect value={feats} onChange={setFeats} options={pool} height={260} />
          </Field>
          <div>
            <Field label="분할 방식">
              <Radio value={method} onChange={setMethod} options={[
                { value: 'random', label: '무작위' }, { value: 'stratified', label: '층화' }, { value: 'group', label: '그룹' }, { value: 'time', label: '시간 순서' }]} />
            </Field>
            {method === 'group' && <Field label="그룹 열" hint="같은 그룹(로트·설비)은 한쪽에만 들어갑니다."><Select value={group} onChange={setGroup} options={c.all} placeholder="선택" /></Field>}
            {method === 'time' && <Field label="시간 열" hint="과거로 학습하고 최근 데이터로 평가합니다."><Select value={time} onChange={setTime} options={c.all} placeholder="선택" /></Field>}
            <Field label={`평가 데이터 비율: ${Math.round(size * 100)}%`}>
              <input type="range" min={0.1} max={0.5} step={0.05} value={size} onChange={(e) => setSize(Number(e.target.value))} />
            </Field>
            <Field label="Random seed"><Num value={seed} onChange={setSeed} step={1} /></Field>
            <Check checked={dropna} onChange={setDropna} label="독립변수 결측 행 모두 삭제 (기본은 파이프라인에서 대체)" />
          </div>
          <div className="explain-box">
            <b>분할 방식 고르기</b>
            <p>· <b>무작위</b>: 행이 서로 독립일 때 기본값</p>
            <p>· <b>층화</b>: 분류 클래스 비율(회귀는 구간 비율)을 학습·평가에 똑같이 유지</p>
            <p>· <b>그룹</b>: 같은 로트·설비·고객의 행이 양쪽에 섞여 성능이 부풀려지는 것을 방지</p>
            <p>· <b>시간 순서</b>: 미래 예측 상황을 그대로 재현</p>
          </div>
        </Grid>
        <div className="actions">
          <RunButton r={r} name="s8_split" params={() => ({ features: feats, method, test_size: size, seed, group_col: group || undefined, time_col: time || undefined, dropna })}
            disabled={!summary.target || !feats.length}>데이터 분할 실행</RunButton>
        </div>
        <RunStatus r={r} />
      </Section>
      <Result rkey="s8" title="분할 결과" />
    </>
  )
}

// ================================================================ 9단계
export function Step9() {
  const { summary } = useProject()
  const r = useRun()
  const c = useColumns()
  const pre = summary.preprocess || {}
  const [scaler, setScaler] = useState(pre.scaler || 'standard')
  const [topn, setTopn] = useState(pre.top_n || 5)
  const [enc, setEnc] = useState(pre.encoding || 'onehot')
  const [asCat, setAsCat] = useState(pre.as_categorical || [])
  const [imp, setImp] = useState(pre.num_impute || 'median')
  const feats = summary.split?.features || []
  const lowCard = c.cols.filter((x) => x.numeric && x.nunique <= 15 && feats.includes(x.name)).map((x) => x.name)
  return (
    <>
      <Need when={!summary.split}>8단계에서 데이터를 먼저 분할하세요.</Need>
      <Section title="표준화 및 더미변수 처리">
        <Grid cols={3}>
          <Field label="연속형 스케일링">
            <Select value={scaler} onChange={setScaler} options={[
              { value: 'standard', label: '표준화 (평균 0, 표준편차 1) — 기준' }, { value: 'robust', label: '강건 스케일 (중앙값·IQR)' },
              { value: 'minmax', label: '0~1 정규화' }, { value: 'none', label: '스케일 없음' }]} />
          </Field>
          <Field label="범주형 인코딩">
            <Radio value={enc} onChange={setEnc} options={[{ value: 'onehot', label: '원-핫(더미)' }, { value: 'target', label: '타깃 인코딩' }]} />
          </Field>
          <Field label="범주별 상위 N개 + 기타" hint="기준 스크립트: 5"><Num value={topn} onChange={setTopn} min={1} max={100} step={1} /></Field>
          <Field label="잔여 결측 대체 (연속형)"><Select value={imp} onChange={setImp} options={[{ value: 'median', label: '중앙값' }, { value: 'mean', label: '평균' }, { value: 'most_frequent', label: '최빈값' }]} /></Field>
          <Field label="숫자지만 범주로 다룰 열" hint="코드값(라인 번호 등)처럼 크기 의미가 없는 숫자 열">
            <MultiSelect value={asCat} onChange={setAsCat} options={lowCard} height={120} />
          </Field>
        </Grid>
        <div className="actions"><RunButton r={r} name="s9_preprocess" params={() => ({ scaler, top_n: topn, encoding: enc, as_categorical: asCat, num_impute: imp })} disabled={!summary.split}>전처리 실행</RunButton></div>
        <RunStatus r={r} />
      </Section>
      <Result rkey="s9" title="전처리 결과" />
    </>
  )
}

// ================================================================ 10단계
export function Step10() {
  const { summary, pid } = useProject()
  const r = useRun()
  const [cat, setCat] = useState(null)
  const [sel, setSel] = useState(null)
  const [tune, setTune] = useState(false)
  const [nIter, setNIter] = useState(15)
  const [folds, setFolds] = useState(5)
  const [metric, setMetric] = useState('')
  const [cw, setCw] = useState(false)
  useEffect(() => {
    if (!summary.target_type) return
    api.get(`/projects/${pid}/models`).then((d) => {
      setCat(d)
      setSel((s) => s || d.models.filter((m) => m.tier === '기준' && m.key !== 'baseline').map((m) => m.key))
      setMetric((m) => m || d.select_metrics[0])
    })
  }, [pid, summary.target_type])
  const models = summary.models || []
  return (
    <>
      <Need when={!summary.split}>8단계에서 데이터를 먼저 분할하세요.</Need>
      {cat && (
        <Section title="모델 선택과 학습 설정" desc="기준선 모델은 항상 함께 학습해 비교합니다.">
          <div className="model-grid">
            {['기준', '심화'].map((tier) => (
              <div key={tier}>
                <h4>{tier === '기준' ? '기준 스크립트 모델' : '심화 모델'}</h4>
                {cat.models.filter((m) => m.tier === tier && m.key !== 'baseline').map((m) => (
                  <Check key={m.key} checked={sel?.includes(m.key)} onChange={(v) => setSel(v ? [...sel, m.key] : sel.filter((x) => x !== m.key))} label={m.name} />
                ))}
              </div>
            ))}
            <div>
              <h4>학습 설정</h4>
              <Field label="교차검증 폴드 수"><Num value={folds} onChange={setFolds} min={2} max={20} step={1} /></Field>
              <Field label="최적 모델 선정 지표"><Select value={metric} onChange={setMetric} options={cat.select_metrics} /></Field>
              <Check checked={tune} onChange={setTune} label="하이퍼파라미터 튜닝 (랜덤 탐색)" />
              {tune && <Field label="모델별 탐색 횟수"><Num value={nIter} onChange={setNIter} min={3} max={200} step={1} /></Field>}
              {summary.target_type !== 'regression' && <Check checked={cw} onChange={setCw} label="클래스 가중치 균형 (불균형 대응)" />}
            </div>
          </div>
          <div className="actions">
            <RunButton r={r} name="s10_train" params={() => ({ models: sel, tune, n_iter: nIter, folds, select_metric: metric, class_weight: cw })} disabled={!sel?.length}>모델 학습 실행</RunButton>
          </div>
          <RunStatus r={r} />
        </Section>
      )}
      {models.length > 0 && (
        <Section title="이후 단계에서 사용할 모델" desc="기본은 자동 선정된 최적 모델입니다. 해석이 쉬운 모델을 쓰고 싶다면 바꿀 수 있습니다.">
          <div className="model-pick">
            {models.map((m) => {
              const main = summary.target_type === 'regression' ? ['RMSE', 'R2_Score'] : ['Accuracy', summary.target_type === 'binary' ? 'AUC' : 'F1_macro']
              return (
                <button key={m.key} className={'pick' + (summary.best_model === m.key ? ' on' : '')} disabled={r.running} onClick={() => r.run('s10_select', { key: m.key })}>
                  <b>{m.name}</b>
                  <span>{main.map((k) => `${k} ${fmt(m.test[k])}`).join(' · ')}</span>
                  {summary.best_model === m.key && <em>사용 중</em>}
                </button>
              )
            })}
          </div>
          <div className="actions">
            {summary.best_model && <a className="button ghost" href={`/api/projects/${pid}/download/model_${summary.best_model}`}>⬇ 선택 모델 파이프라인(.joblib)</a>}
          </div>
        </Section>
      )}
      <Result rkey="s10_select" />
      <Result rkey="s10" title="학습·평가 결과" />
    </>
  )
}

// ================================================================ 11~12단계
export function Step11() {
  const { summary } = useProject()
  const r = useRun()
  const c = useColumns()
  const [lc, setLc] = useState(false)
  const [kinds, setKinds] = useState(['perm', 'model', 'shap', 'pdp'])
  const [pdp, setPdp] = useState([])
  const [model, setModel] = useState('')
  const feats = summary.split?.features || []
  const tog = (k) => (v) => setKinds(v ? [...kinds, k] : kinds.filter((x) => x !== k))
  return (
    <>
      <Need when={!summary.best_model}>10단계에서 모델을 먼저 학습하세요.</Need>
      <Section title="모델링 결과 비교 및 시각화" actions={<RunButton r={r} name="s11_compare" params={() => ({ learning_curve: lc })} disabled={!summary.best_model}>비교 시각화 실행</RunButton>}>
        <Check checked={lc} onChange={setLc} label="학습곡선 포함 (데이터를 더 모으면 좋아질지 진단, 시간 소요)" />
      </Section>
      <Result rkey="s11_compare" title="모델 비교·진단" />
      <Section title="설명 가능성 (심화)" desc="모델이 어떤 변수로 어떻게 예측하는지 보여 줍니다.">
        <Grid cols={3}>
          <Field label="설명할 모델">
            <Select value={model} onChange={setModel} options={(summary.models || []).filter((m) => m.key !== 'baseline').map((m) => ({ value: m.key, label: m.name }))} placeholder="최적 모델" />
          </Field>
          <Field label="분석 항목">
            <Check checked={kinds.includes('perm')} onChange={tog('perm')} label="순열 중요도" />
            <Check checked={kinds.includes('model')} onChange={tog('model')} label="모델 내장 중요도·계수" />
            <Check checked={kinds.includes('shap')} onChange={tog('shap')} label="SHAP" />
            <Check checked={kinds.includes('pdp')} onChange={tog('pdp')} label="부분의존도 PDP·ICE" />
          </Field>
          <Field label="PDP 변수 (비우면 중요도 상위 4개)"><MultiSelect value={pdp} onChange={setPdp} options={feats} height={140} /></Field>
        </Grid>
        <div className="actions"><RunButton r={r} name="s11_explain" params={() => ({ kinds, pdp_features: pdp, model: model || undefined })} disabled={!summary.best_model || !kinds.length}>설명 분석 실행</RunButton></div>
        <RunStatus r={r} />
      </Section>
      <Result rkey="s11_explain" title="설명 가능성" />
    </>
  )
}
