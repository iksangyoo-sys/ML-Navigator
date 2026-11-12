import { Step1, Step2, Step3, Step4 } from './Data'
import { Step5, Step6, Step7 } from './Prep'
import { Step8, Step9, Step10, Step11 } from './Model'
import { Step13, Step14, Step15, Step16 } from './Advanced'
import Report from './Report'

export const STEPS = [
  { id: 1, num: 1, title: '데이터 불러오기', comp: Step1, desc: 'CSV·XLSX·Parquet 파일을 불러오고 자동 프로파일로 열 타입과 역할을 확인합니다.' },
  { id: 2, num: 2, title: '데이터 결합하기', comp: Step2, desc: '여러 파일을 행 또는 열 방향으로 결합하고 키 관계·행 증식·미매칭을 진단합니다.' },
  { id: 3, num: 3, title: '파생변수 생성하기', comp: Step3, desc: '구간화·더미·Tukey·사칙연산·변환 등으로 새 변수를 만듭니다. 모두 레시피로 기록되어 신규 데이터에 재적용됩니다.' },
  { id: 4, num: 4, title: '종속변수 분포확인', comp: Step4, desc: '예측할 종속변수를 정하고 분포·정규성·클래스 불균형을 확인합니다.' },
  { id: 5, num: 5, title: '결측치 처리', comp: Step5, desc: '결측 패턴과 타깃 관련성을 진단하고 삭제·대체·지시변수로 처리합니다.' },
  { id: 6, num: 6, title: '이상치 처리', comp: Step6, desc: 'IQR·Z·MAD·백분위·다변량 방법으로 이상치를 찾고 제거·윈저화·대체합니다.' },
  { id: 7, num: 7, title: '탐색적 분석', comp: Step7, desc: '변수 간 관계를 산점도·상관·ANOVA·교차표·VIF·상호정보량으로 탐색합니다.' },
  { id: 8, num: 8, title: '데이터 분할', comp: Step8, desc: '학습/평가 데이터를 나누고 분할이 치우치지 않았는지 확인합니다.' },
  { id: 9, num: 9, title: '모델링 전처리', comp: Step9, desc: '표준화·인코딩 파이프라인을 학습 데이터 기준으로 구성합니다.' },
  { id: 10, num: 10, title: '모델 학습 및 평가', comp: Step10, desc: '여러 모델을 교차검증·튜닝으로 비교하고 최적 모델을 고릅니다.' },
  { id: 11, num: '11·12', title: '모델 비교 및 시각화', comp: Step11, desc: '예측 성능을 시각화하고 순열 중요도·SHAP·PDP로 모델을 설명합니다.' },
  { id: 13, num: 13, title: '직접 입력 예측', comp: Step13, desc: '조건을 입력해 예측하고 What-if·민감도로 반응을 확인합니다.' },
  { id: 14, num: 14, title: 'DoWhy 인과분석', comp: Step14, desc: 'Model → Identify → Estimate → Refute로 변수의 인과효과를 추정합니다.' },
  { id: 15, num: 15, title: 'Counterfactual XAI', comp: Step15, desc: '목표 예측을 만족하는 최소 변경 조건과 최적 운전 조건을 찾습니다.' },
  { id: 16, num: 16, title: '신규 데이터 예측', comp: Step16, desc: '새 데이터에 레시피와 모델을 적용해 예측하고 드리프트·성능을 점검합니다.' },
  { id: 'report', num: '★', title: '보고서', comp: Report, desc: '단계별 분석 보고서와 경영 요약을 만들고 PDF로 저장합니다.' },
]
