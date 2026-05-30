@echo off
title SafeGuard AI - Surveillance System Launcher

:: Force working directory to be the batch file's folder
cd /d "%~dp0"

echo ──────────────────────────────────────────────────────────
echo 🛡️  Starting SafeGuard AI Backend Server...
echo ──────────────────────────────────────────────────────────
echo.

:: Check if virtual environment exists
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment (.venv) not found in: %CD%
    echo Please make sure you are in the project root and run setup.
    pause
    exit /b 1
)

:: Start Flask backend in a separate background window
start "SafeGuard AI Backend" cmd /k ".venv\Scripts\python.exe backend\app.py"

echo.
echo Waiting for backend server to initialize...
timeout /t 5 /nobreak > nul

echo.
echo ──────────────────────────────────────────────────────────
echo 📊 Opening Web Dashboard...
echo ──────────────────────────────────────────────────────────
start http://127.0.0.1:5000/

echo.
echo SafeGuard AI is now running!
echo - Keep the backend terminal window open to keep the system active.
echo - You can access the dashboard at http://127.0.0.1:5000/
echo.
pause
