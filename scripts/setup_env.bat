@echo off
REM Canonical install lines: README §7 step 2 / §7b step 2 (requirements + editable dev + voice).
cd /d "%~dp0.."
where py >nul 2>nul && (set PY=py -3) || (set PY=python)
%PY% -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e ".[dev,voice]"
echo.
echo Done. Activate with:  .venv\Scripts\activate.bat
echo Copy env templates: copy .env.example .env  (then edit)
pause
