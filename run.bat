@echo off
REM ============================================================
REM  ExplainMyXray — Start API + Frontend (Windows)
REM ============================================================

echo ====================================
echo   ExplainMyXray Local Dev Server
echo ====================================
echo.

REM Activate venv if it exists
if exist "%~dp0venv\Scripts\activate.bat" (
    call "%~dp0venv\Scripts\activate.bat"
    echo [OK] Virtual environment activated.
) else (
    echo [INFO] No venv found — using system Python.
)

REM Check dependencies
python -c "import streamlit, fastapi, uvicorn" 2>nul
if errorlevel 1 (
    echo [INSTALL] Installing dependencies...
    pip install -r "%~dp0requirements.txt" --quiet
)

echo.
echo Starting API server on http://localhost:8000 ...
start "ExplainMyXray API" cmd /c "cd /d %~dp0 && python app/api.py"

REM Wait for API to be ready
timeout /t 4 /nobreak >nul

echo Starting Streamlit frontend on http://localhost:8501 ...
start "ExplainMyXray UI" cmd /c "cd /d %~dp0 && streamlit run app/frontend.py --server.port 8501"

echo.
echo ====================================
echo   Both servers starting!
echo   API:      http://localhost:8000
echo   Frontend: http://localhost:8501
echo ====================================
echo.
echo Open http://localhost:8501 in your browser.
echo Close the two server windows to stop.
pause
