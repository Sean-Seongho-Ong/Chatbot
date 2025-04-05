@echo off
chcp 65001 > nul REM Set UTF-8 encoding (for displaying Korean messages)
setlocal enabledelayedexpansion REM Required for using !errorlevel!

echo --- Script Start ---
pause REM Confirm start

echo Terminating processes using port 3000...

REM Find and terminate the process PID using port 3000 in LISTENING state
set "PID_FOUND="
for /f "tokens=5" %%a in ('netstat -aon ^| findstr /L ":3000" ^| findstr "LISTENING"') do (
    set "PID_FOUND=%%a"
    echo Found PID: !PID_FOUND!
    echo Attempting to terminate process...
    taskkill /F /PID !PID_FOUND!
    if !errorlevel! equ 0 (
        echo Successfully terminated process !PID_FOUND!.
        ping 127.0.0.1 -n 2 > nul REM Wait briefly
    ) else (
        echo Failed to terminate process !PID_FOUND!.
    )
    set PID_FOUND= REM Prevent next iteration (may not be needed)
)

if not defined PID_FOUND (
    echo No process using port 3000.
)

echo --- Port 3000 cleanup complete ---
pause REM Confirm after port cleanup

REM --- Actual chatbot start logic begins here ---
echo.
echo Starting ChatBot application...
echo.

echo Checking backend server status...
curl -s http://localhost:8000/health > nul
if %errorlevel% equ 0 (
    echo Backend server is already running.
) else (
    echo Starting backend server...
    cd /d "%~dp0backend"
    if errorlevel 1 (
      echo ERROR: Failed to change to backend directory!
      pause
      goto EndScript
    )
    set "PYTHONIOENCODING=utf-8"
    echo Executing backend start command: python -m uvicorn ...
    start "Backend Server" cmd /c "python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload --log-level debug"
    cd /d "%~dp0"
    echo Waiting 45 seconds for backend server to initialize...
    ping 127.0.0.1 -n 46 > nul
)

echo --- Backend processing complete ---
pause REM Confirm after backend processing

echo Starting frontend server...
cd /d "%~dp0frontend"
if errorlevel 1 (
  echo ERROR: Failed to change to frontend directory!
  pause
  goto EndScript
)
echo Executing frontend start command: npm start
start "Frontend Server" cmd /c "npm start"
cd /d "%~dp0"

echo --- Frontend start complete ---
pause REM Confirm after frontend start

echo Both servers have started.
echo The web browser will automatically open at http://localhost:3000.
echo Close these windows to terminate the application.

REM Command to execute on exit
echo.
echo Press any key to exit...
pause > nul

REM Exit processing
echo Terminating servers...
REM Attempt to terminate uvicorn-related processes in addition to python, node
taskkill /F /IM uvicorn.exe /T 2>nul
taskkill /F /IM python.exe /T 2>nul
taskkill /F /IM node.exe /T 2>nul
taskkill /F /IM conhost.exe /FI "WINDOWTITLE eq Backend Server" /T 2>nul REM Close window by title
taskkill /F /IM conhost.exe /FI "WINDOWTITLE eq Frontend Server" /T 2>nul
echo Servers have been terminated.

:EndScript
endlocal REM Release setlocal
echo --- Script End --- 