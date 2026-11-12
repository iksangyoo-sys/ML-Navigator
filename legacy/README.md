# 공정 데이터 분석

Flask 기반 정형 데이터 분석 작업공간입니다. 요구사항은 PRD.md를 참고하세요.

## 로컬 실행

Windows에서 Python 3.12를 지정해 실행합니다. 이 작업 폴더에 Python 3.12용 `.vendor`가 준비되어 있고 `python --version`이 3.12.x이면 `python app.py`로 바로 시작할 수 있습니다. `py` 런처가 3.12를 찾으면 `py -3.12 app.py`도 사용할 수 있습니다.

새 작업 폴더에서는 Python 3.12로 가상환경을 만듭니다.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python app.py
```

브라우저에서 http://127.0.0.1:5000 을 엽니다. `py -3.12`가 설치된 Python을 찾지 못하면 `python --version`을 확인한 뒤 Python 3.12의 `python.exe`를 직접 사용하세요. 프런트엔드 문법 검사는 Node.js가 설치된 환경에서 `npm run build`로 실행할 수 있습니다.

## 분석 흐름

1. 작업공간에서 data 폴더의 파일을 선택하거나 업로드합니다. 파일 두 개 이상을 결합할 때는 화면 목록 순서에서 첫 파일이 기준입니다. 동일 스키마의 배치를 이어 붙일 때는 data.combine에 append를 지정합니다.
2. 작업 유형과 파일을 고르고 분석 목적, 관측 단위, 독립·대응 여부를 적습니다. **GPT-4o mini 기법 추천**을 누르면 방법별 적용 근거와 가정을 확인할 수 있습니다. **옵션에 적용**으로 기법과 변수 후보를 채운 뒤 화면에서 수정하고 실행합니다. 예측 문제에서는 개발용 데이터로 추천을 받고 예측 시점에 사용 가능한 특징을 명시합니다.
3. 결과 화면에서 차트, 지표, 전처리 이력 및 CSV를 검토합니다. 승인한 모델 ID는 배치 모니터링에 사용합니다. 승인한 DOE 설계의 design.csv에서 run_id를 보존해 측정값을 기록한 후 DOE 결과 작업에 입력합니다.
4. 실행 결과에서 **GPT-4o mini 보고서 초안**을 눌러 집계 기반 문안을 생성하고 편집·승인합니다. 경영 보고서에서 승인된 실행의 **GPT-4o mini 종합 초안**을 생성할 수 있습니다. 보고서 확정본에는 당시 문안과 근거 목록이 저장됩니다.

OpenAI 키는 서버에서 `OPENAI_API_KEY`를 읽습니다. Windows에서는 사용자 또는 시스템 환경변수도 확인합니다. 추천에는 최대 2,000행에서 계산한 열 수준 집계와 입력한 목적을 사용합니다. 보고서에는 실행 지표와 사용자가 승인한 문안을 사용합니다. API 요청 없이 화면에서 기법을 직접 선택할 수 있습니다. 가상환경을 이미 만들었다면 `requirements.txt`를 다시 설치해 OpenAI SDK를 추가하세요.

## 옵션 예시

- 데이터 읽기: data.read에 파일명별 encoding, separator, sheet(시트명 또는 0 기반 번호), columns(Parquet)를 지정합니다.
- 결합: data.keys, join(left/inner/outer), cardinality(one_to_one, many_to_one 등), max_join_rows로 기대 구조를 설정합니다.
- 전처리: data.rules 배열의 각 항목에 column과 operation을 지정합니다. numeric, date, unit(factor/offset), range(min/max), allowed(values), drop_missing, deduplicate(keys/keep), ratio·difference·sum·product(other/output), log(output)를 지원합니다. 범위·허용값 규칙의 action은 flag, mask, exclude입니다.
- EDA: options.chart는 histogram, frequency, scatter, trend, ecdf를 선택합니다. columns, x, y, group, filters, sample, bins, top, crosstab, correlation, test(kruskal)를 설정할 수 있습니다.
- 모델: options.models는 linear, tree, forest, boosting을 선택할 수 있으며 baseline을 항상 비교합니다. split은 random, stratified, group, time, group_time입니다. clip(iqr/mad), clip_factor, power(yeo-johnson/box-cox)는 학습 폴드에서 적합됩니다. group/time, segments, exclude는 열 이름으로 입력합니다. 선택 지표는 scikit-learn scoring 이름을 사용합니다.
- SPC: options.chart는 imr, xbar_r, xbar_s, p, u를 선택합니다. value, time, start, end와 필요 시 group, denominator, lsl/usl을 설정합니다. Phase I 기간을 포함한 데이터에 Phase II 배치를 추가해 기준 한계를 고정 적용합니다.
- DOE: design은 full, fractional, ccd, box_behnken입니다. 부분 요인 예시는 generators에 {"C":["A","B"]}를 지정합니다. 실적 CSV에는 run_id와 반응열, 필요한 경우 실제 요인값을 같은 요인명으로 기록합니다. 별칭이 존재할 때는 options.terms에 추정할 항을 명시합니다. 교호작용 이름은 "A × B", 제곱항은 "A²"입니다.

## Render와 저장

render.yaml은 무료 Flask 서비스와 임시 저장소를 설정합니다. 업로드와 실행 이력은 재시작이나 재배포 시 사라질 수 있습니다. Render 환경 변수 화면에서 자동 생성된 ANALYSIS_PASSWORD를 확인해 로그인에 사용합니다. 기본 HTTPS 주소는 Render 환경에서 자동 감지하며, 별도 도메인에는 PUBLIC_ORIGIN을 설정할 수 있습니다. 무료 인스턴스에서는 소규모 데이터로 분석을 시작하세요.

## 확인과 현재 범위

```powershell
npm run build
python tests/smoke.py
```

현재 구현은 pandas의 메모리 내 처리입니다. 1GB는 입력 바이트 상한이며 그 크기에서의 모든 분석 성공을 보장하는 벤치마크는 아직 없습니다. 실행 중 취소는 결과 확정을 막지만 진행 중인 모델 계산이 반환될 때까지 워커를 점유할 수 있습니다. 재시작 중이던 작업은 중단으로 기록되고 재실행할 수 있습니다. 비정형 분석은 후속 출시 범위입니다. 별도 레이블 파일 조인, 고급 DOE 시각화·복수 반응 최적화, 자동 모델 탐색·임계값 보정은 아직 구현하지 않았습니다.
