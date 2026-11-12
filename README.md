# 빅데이터 분석 자동화 웹

데이터가 바뀌어도 `M6_종합예측_counterfactual_dowhy.py`의 16단계 분석 흐름을 그대로 따르면서, 단계마다 더 깊은 분석을 제공하는 웹입니다. 요구사항은 [PRD.md](PRD.md)를 참고하세요.

## 실행 (Windows 로컬)

`run_local.bat`을 더블클릭하면 가상환경 준비, 패키지 설치, 화면 빌드 후 브라우저에서 http://127.0.0.1:5000 이 열립니다.

직접 실행할 때:

```powershell
.\.venv\Scripts\python -m pip install -r requirements.txt
cd frontend; npm install; npm run build; cd ..
.\.venv\Scripts\python -m backend.app
```

- `data` 폴더에 넣은 파일은 1단계에서 "서버 data 폴더" 버튼으로 바로 불러올 수 있습니다.
- 프로젝트·결과·모델은 `instance/projects`에 저장됩니다(`ANALYSIS_HOME`으로 변경 가능).
- 로그인 비밀번호는 환경변수 `APP_PASSWORD`로 설정합니다. 설정하지 않으면 로컬에서는 로그인 없이 열립니다.
- `OPENAI_API_KEY`가 있으면 결과 패널에 **AI 해설** 버튼이 나타납니다(선택 기능, 모델은 `OPENAI_MODEL`, 기본 gpt-4o-mini).

화면 개발 시에는 백엔드를 켠 상태에서 `cd frontend; npm run dev`로 http://localhost:5173 을 엽니다.

## 16단계 흐름

| 단계 | 기준 스크립트 기능 | 추가된 심화 분석 |
|---|---|---|
| 1 불러오기 | 업로드, 인코딩 판별, 기술통계 | 서버 data 폴더, 시트 선택, Parquet, 자동 프로파일, 열 역할 |
| 2 결합 | 행·열 결합, 공통 열 키 | 키 직접 지정, 키 관계(1:1·N:N) 검증, 행 증식·미매칭 진단 |
| 3 파생변수 | binning, dummy, Tukey, 사칙연산, transform | Box-Cox·Yeo-Johnson, 날짜 분해, 그룹 집계, 조건식, 범주 병합, 수식, Tukey 왜도 비교, 레시피 재적용 |
| 4 종속변수 | 분포·막대·파이 | Q-Q, 정규성 검정, 변환 제안, 불균형 진단, 양성 클래스 지정 |
| 5 결측 | 삭제, 평균·중앙·최빈 | 결측 패턴, 결측-타깃 관련성, KNN·반복 대체, 지시변수 |
| 6 이상치 | IQR, 제거·대체·직접 범위 | Z·MAD·백분위, 방법 비교, 윈저화, Isolation Forest·마할라노비스 |
| 7 EDA | 산점도, 박스플롯, 상관, ANOVA | LOWESS, Tukey HSD, η², Kruskal, 편상관, 카이제곱·Cramér's V, VIF, 상호정보량, 산점도 행렬, 시간 추이, 관련 변수 요약 |
| 8 분할 | 무작위·층화 | 그룹·시간 분할, 분포 비교(KS·표준화 평균차) |
| 9 전처리 | 표준화, 상위 5 + 원-핫 | 스케일러 선택, 타깃 인코딩, 학습형 처리 파이프라인 |
| 10 학습 | 7개 모델, 학습·평가 지표 | 기준선, 심화 모델 6종, K-겹 교차검증, 튜닝, 과적합 진단, 모델 수동 지정 |
| 11·12 비교 | 실제 vs 예측, ROC | 잔차 진단, 혼동행렬, PR·보정 곡선, 임계값 최적화, 리프트, 학습곡선, 순열 중요도, SHAP, PDP·ICE |
| 13 직접 입력 | 입력 폼, 예측 | 파생변수 자동 계산, 예측구간, 개별 SHAP, What-if, 토네이도 민감도 |
| 14 DoWhy | 4단계, 반박 3종, OLS 대체 | 원 단위 효과, DAG 지정, 반박 5종, OLS(HC3)·DML 비교, 변수 일괄 효과 vs 예측 중요도, 이질적 효과 |
| 15 Counterfactual | 무작위 탐색 | 범위·방향·비용 제약, 유전 알고리즘, 다양성, 근접 실측 검증, 인과 방향 검증, 일괄 개선안, 최적 조건 |
| 16 신규 예측 | 예측, 성능평가 | 스키마 검증, 레시피 재적용, PSI·KS 드리프트, 학습 대비 성능 비교 |

모든 분석은 데이터 누수를 막는 방식으로 동작합니다. 결측 대체·이상치 기준은 분할 후 학습 데이터로만 계산하고, 최적 모델은 교차검증으로 선택하며, 평가 데이터는 최종 확인에만 씁니다.

## 검증

```powershell
.\.venv\Scripts\python -m tests.test_flow            # 회귀·이진·다중 분류 3종으로 1~16단계 전체 실행
```

## Render 배포

`render.yaml`로 배포합니다. 결과를 보존하려면 영속 디스크가 필요해 유료(Starter) 플랜으로 설정했습니다. 배포 후 Render 환경변수 화면에서 자동 생성된 `APP_PASSWORD`를 확인해 로그인하세요. 작업 상태를 메모리에 두므로 gunicorn 워커는 1개를 유지해야 합니다.

## 폴더 구조

- `backend/` Flask API, 작업 실행기, 보고서
- `backend/analysis/` 단계별 분석 엔진 (dataio, merge, features, recipe, target, missing, outliers, eda, modeling, evaluate, predict, causal, counterfactual, newdata)
- `frontend/` React + Vite 화면
- `tests/test_flow.py` 종단 테스트
- `legacy/` 이전 Codex 코드(참고용 보관)
