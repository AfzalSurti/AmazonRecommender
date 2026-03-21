@echo off
setlocal

cd /d "%~dp0"
set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
  echo [ERROR] Virtual environment Python not found at:
  echo %PYTHON_EXE%
  echo Create the virtual environment and install dependencies first.
  exit /b 1
)

echo Starting Amazon Web Scrapper app...
echo URL: http://127.0.0.1:5000
echo Press Ctrl+C to stop.
echo.

"%PYTHON_EXE%" app_symentic.py

endlocal
