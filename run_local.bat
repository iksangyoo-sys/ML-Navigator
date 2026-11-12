@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo [1/3] Python 가상환경을 만듭니다...
  py -3.12 -m venv .venv || python -m venv .venv
)
echo [2/3] 패키지를 확인합니다...
.venv\Scripts\python.exe -m pip install -q -r requirements.txt
if not exist frontend\dist\index.html (
  echo [3/3] 화면을 빌드합니다...
  pushd frontend
  call npm install
  call npm run build
  popd
)
start "" http://127.0.0.1:5000
.venv\Scripts\python.exe -m backend.app
pause
