@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "LOG=%~dp0setup.log"
set "PYCMD="
set "RC=0"

REM --- find a Python 3.11+ (no parenthesised blocks: they break when the folder name contains brackets) ---
py -3.12 -c "import sys" >nul 2>&1
if not errorlevel 1 set "PYCMD=py -3.12"
if defined PYCMD goto :found
py -3 -c "import sys; sys.exit(0 if sys.version_info[:2]>=(3,11) else 1)" >nul 2>&1
if not errorlevel 1 set "PYCMD=py -3"
if defined PYCMD goto :found
python -c "import sys; sys.exit(0 if sys.version_info[:2]>=(3,11) else 1)" >nul 2>&1
if not errorlevel 1 set "PYCMD=python"
if defined PYCMD goto :found

echo ERROR: no usable Python 3.11+ found. Tried: py -3.12, py -3, python.
echo If Python is installed, reinstall it from python.org and tick "Add python.exe to PATH".
echo The Microsoft Store python.exe alias does not count.
echo ERROR: no usable Python 3.11+ found> "%LOG%"
set "RC=9009"
goto :fail

:found
echo Using Python launcher command: %PYCMD%
%PYCMD% scripts\bootstrap.py
set "RC=%errorlevel%"
if not "%RC%"=="0" goto :fail
echo.
echo Setup completed successfully. Next: start_api.bat in one window, start_ui.bat in another.
pause
exit /b 0

:fail
echo.
echo ======================================================
echo SETUP FAILED, exit code %RC%.
echo The failing command and its full output are shown above and saved in:
echo   %LOG%
echo ======================================================
pause
exit /b 1
