@echo off
REM Start both services in full mode (real ML models).
REM
REM On first run, builds the embedding index (~5-10 min).
REM Close the terminal windows to stop services.

cd /d %~dp0

REM Check if embeddings exist, build if not
if not exist "data\embeddings.npy" (
    echo.
    echo ============================================
    echo  First run: building embedding index...
    echo  This downloads CLIP weights + part images.
    echo  Takes ~5-10 minutes. Please wait.
    echo ============================================
    echo.
    python scripts/build_embeddings.py
    if errorlevel 1 (
        echo.
        echo ERROR: Embedding build failed. Check the output above.
        pause
        exit /b 1
    )
    echo.
    echo Embeddings built successfully!
    echo.
)

echo Starting Python API on http://localhost:8000 ...
start "LEGO API" cmd /k "cd /d %~dp0 && python -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload"

timeout /t 5 /nobreak >nul

echo Starting Next.js frontend on http://localhost:3000 ...
start "LEGO Web" cmd /k "cd /d %~dp0\web && npm run dev"

echo.
echo ============================================
echo  Services starting.
echo  Open http://localhost:3000 in your browser.
echo  Close the terminal windows to stop.
echo ============================================
pause
