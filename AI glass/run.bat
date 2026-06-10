@echo off
cd /d "%~dp0"
echo Starting AI Glass UI at http://127.0.0.1:7860
python app.py
if errorlevel 1 (
  echo.
  echo If "python" not found, try: D:\anaconda\python.exe app.py
  pause
)
