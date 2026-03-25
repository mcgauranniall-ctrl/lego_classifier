@echo off
cd /d "%~dp0"
echo.
echo   LEGO Piece Identifier
echo   =====================
echo   Starting server...
echo   Open http://localhost:8000 in your browser
echo   Press Ctrl+C to stop
echo.
python brickognize/server.py
pause
