@echo off
REM Double-click to open the exudate detector in the browser.
cd /d "%~dp0"
if not exist .venv (
  python -m venv .venv
  .venv\Scripts\python -m pip install -r requirements.txt
)
.venv\Scripts\python app.py
