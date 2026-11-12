@echo off
cd /d "%~dp0"
set "PYTHONPATH=%CD%"
set "PYTHONUTF8=1"
if not exist ".venv\Scripts\python.exe" (echo Run setup.bat first & pause & exit /b 1)
".venv\Scripts\python.exe" -m streamlit run frontend\app.py
