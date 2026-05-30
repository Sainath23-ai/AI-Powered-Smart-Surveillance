@echo off
chcp 65001 > nul
title SafeGuard AI - Surveillance System Launcher
cd /d "%~dp0"

echo.
echo  ================================================================
echo   SafeGuard AI  --  Surveillance System Launcher
echo  ================================================================
echo.

:: Step 1: Verify virtual environment
if not exist ".venv\Scripts\python.exe" (
    echo  [ERROR] Virtual environment .venv not found.
    echo          Run setup first, then try again.
    echo.
    pause
    exit /b 1
)

:: Step 2: Launch Flask backend in its own window
echo  [1/3] Starting Flask backend on http://127.0.0.1:5000 ...
start "SafeGuard AI Backend" cmd /k ".venv\Scripts\python.exe backend\app.py"

:: Step 3: Poll /api/ping until backend is ready (max 30 seconds)
echo  [2/3] Waiting for backend to become ready...
set TRIES=0

:WAIT_LOOP
timeout /t 1 /nobreak > nul
set /a TRIES=TRIES+1

curl -s -o nul -w "%%{http_code}" http://127.0.0.1:5000/api/ping 2>nul | findstr "200" > nul
if not errorlevel 1 goto BACKEND_READY

if %TRIES% GEQ 30 (
    echo.
    echo  [WARNING] Backend did not respond after 30s. Opening anyway...
    goto BACKEND_READY
)
goto WAIT_LOOP

:BACKEND_READY
echo  [3/3] Backend ready! Opening web dashboard...
echo.

:: Step 4: Open dashboard in default browser
start "" "http://127.0.0.1:5000/"

echo.
echo  ================================================================
echo   SafeGuard AI is running!
echo   Dashboard : http://127.0.0.1:5000/
echo   Keep the backend window open to stay active.
echo  ================================================================
echo.
pause
