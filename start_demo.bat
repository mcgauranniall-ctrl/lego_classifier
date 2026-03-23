@echo off
REM Start both services in demo mode for end-to-end testing (Windows).
REM
REM Usage:
REM   start_demo.bat          — starts Python API (port 8000) + Next.js (port 3000)
REM
REM Demo mode returns realistic fake detections without loading YOLO+CLIP models.
REM Close the terminal windows to stop services.

set DEMO_MODE=1

echo Starting Python API (demo mode) on http://localhost:8000 ...
start "LEGO API" cmd /k "cd /d %~dp0 && python run_demo_api.py"

timeout /t 3 /nobreak >nul

echo Starting Next.js frontend on http://localhost:3000 ...
start "LEGO Web" cmd /k "cd /d %~dp0\web && npm run dev"

echo.
echo Services starting. Open http://localhost:3000 in your browser.
echo Close the terminal windows to stop services.
pause
