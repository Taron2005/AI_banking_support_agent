@echo off
cd /d "%~dp0..\frontend-react"
if not exist "node_modules" (
  echo Running npm install ...
  call npm install
)
echo Opening browser at http://127.0.0.1:5173 ...
start "" "http://127.0.0.1:5173"
call npm run dev
pause
