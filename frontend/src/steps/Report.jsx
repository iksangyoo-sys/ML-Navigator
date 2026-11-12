import { useEffect, useMemo, useState } from 'react'
import { api } from '../api'
import { Field, Grid, Section, useProject } from '../components/ui'
import { STEPS } from './index'

const LABELS = {
  s1: '데이터 확인', s2: '결합 결과', s2_keys: '결합 키 진단', s3: '파생변수', s3_tukey: 'Tukey 비교', s4: '종속변수 분포', s5: '결측치',
  s6: '이상치', s6_multi: '다변량 이상치', s7_summary: 'EDA 요약', s7_scatter: '산점도', s7_box: '박스플롯', s7_corr: '상관분석', s7_partial: '편상관',
  s7_anova: 'ANOVA', s7_classtarget: '분류 타깃 검정', s7_crosstab: '교차표', s7_vif: 'VIF', s7_mi: '상호정보량', s7_matrix: '산점도 행렬', s7_trend: '시간 추이',
  s8: '데이터 분할', s9: '전처리', s10: '모델 학습', s10_select: '모델 지정', s11_compare: '모델 비교', s11_explain: '설명 가능성', s13: '직접 입력 예측',
  s13_whatif: 'What-if', s13_sens: '민감도', s14: 'DoWhy', s14_batch: '일괄 인과효과', s14_hetero: '이질적 효과', s15: 'Counterfactual',
  s15_batch: '일괄 Counterfactual', s15_opt: '최적 조건', s16: '신규 데이터 예측',
}

export default function Report() {
  const { pid, summary } = useProject()
  const keys = useMemo(() => Object.keys(summary.results).sort((a, b) => summary.results[a].step - summary.results[b].step || a.localeCompare(b)), [summary.results])
  const [sel, setSel] = useState(keys.filter((k) => !['s10_select', 's3_tukey', 's2_keys'].includes(k)))
  const [f, setF] = useState(null)
  const [busy, setBusy] = useState(false)
  useEffect(() => {
    api.get(`/projects/${pid}/report/executive-fields`).then(setF)
  }, [pid])
  const openSteps = () => window.open(`/api/projects/${pid}/report/steps?keys=${sel.join(',')}`, '_blank')
  const openExec = async () => {
    setBusy(true)
    const w = window.open('', '_blank')
    const { html } = await api.post(`/projects/${pid}/report/executive`, f)
    w.document.open()
    w.document.write(html)
    w.document.close()
    setBusy(false)
  }
  const set = (k) => (e) => setF({ ...f, [k]: e.target.value })
  return (
    <>
      <Section title="단계별 분석 보고서 (HTML·PDF)" desc="포함할 결과를 고르고 보고서를 열면, 새 창의 'PDF로 저장' 버튼으로 PDF를 만들 수 있습니다."
        actions={<button className="primary" onClick={openSteps} disabled={!sel.length}>단계별 보고서 열기</button>}>
        {!keys.length ? <p className="muted">아직 실행한 결과가 없습니다.</p> : (
          <div className="report-keys">
            {STEPS.filter((s) => typeof s.id === 'number').map((s) => {
              const ks = keys.filter((k) => summary.results[k].step === s.id)
              if (!ks.length) return null
              return (
                <div key={s.id}>
                  <b>{s.num}. {s.title}</b>
                  {ks.map((k) => (
                    <label key={k} className="check">
                      <input type="checkbox" checked={sel.includes(k)} onChange={(e) => setSel(e.target.checked ? [...sel, k] : sel.filter((x) => x !== k))} />
                      {LABELS[k] || k}{summary.results[k].stale && <span className="badge warn">재실행 필요</span>}
                    </label>
                  ))}
                </div>
              )
            })}
          </div>
        )}
      </Section>
      {f && (
        <Section title="경영 요약 (HTML·PDF)" desc="의사결정, 기준 대비 KPI, 불확실성, 권고, 담당과 기한을 담습니다. 모델 성능·인과효과·최적 조건은 실행 결과에서 자동으로 붙습니다."
          actions={<button className="primary" onClick={openExec} disabled={busy}>{busy ? '생성 중…' : '경영 요약 열기'}</button>}>
          <Grid cols={2}>
            <Field label="의사결정 질문"><textarea rows={2} value={f.question || ''} onChange={set('question')} /></Field>
            <Field label="핵심 결론"><textarea rows={2} value={f.conclusion || ''} onChange={set('conclusion')} placeholder="예: 소성온도를 10℃ 낮추면 탭밀도가 평균 0.05 높아질 것으로 추정" /></Field>
            <Field label="기준 대비 성과(KPI)"><textarea rows={2} value={f.kpi || ''} onChange={set('kpi')} /></Field>
            <Field label="불확실성·한계"><textarea rows={2} value={f.uncertainty || ''} onChange={set('uncertainty')} /></Field>
            <Field label="권고 사항"><textarea rows={3} value={f.recommendation || ''} onChange={set('recommendation')} /></Field>
            <div>
              <Field label="담당"><input value={f.owner || ''} onChange={set('owner')} /></Field>
              <Field label="기한"><input type="date" value={f.due || ''} onChange={set('due')} /></Field>
            </div>
          </Grid>
        </Section>
      )}
      <Section title="데이터·모델 내려받기">
        <div className="actions left">
          <a className="button ghost" href={`/api/projects/${pid}/download/merged`}>⬇ 결합 데이터 CSV</a>
          <a className="button ghost" href={`/api/projects/${pid}/download/working`}>⬇ 작업 데이터(파생변수 포함) CSV</a>
          <a className="button ghost" href={`/api/projects/${pid}/download/view`}>⬇ 결측·이상치 처리 반영 CSV</a>
          {summary.best_model && <a className="button ghost" href={`/api/projects/${pid}/download/model_${summary.best_model}`}>⬇ 모델 파이프라인 .joblib</a>}
        </div>
      </Section>
      {summary.history?.length > 0 && (
        <Section title="처리 이력">
          <ol className="recipe">
            {[...summary.history].reverse().map((h, i) => (
              <li key={i}><span><b>{h.step}단계</b> · {h.text}</span><span className="muted small">{new Date(h.time * 1000).toLocaleString('ko-KR')}</span></li>
            ))}
          </ol>
        </Section>
      )}
    </>
  )
}
