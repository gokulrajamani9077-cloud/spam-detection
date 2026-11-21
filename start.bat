@echo off
title SMS SHIELD - AI-Powered Spam Detection System
color 0A
cls

echo ======================================================================
echo           SMS SHIELD - AI-POWERED SPAM DETECTION SYSTEM
echo ======================================================================
echo.
echo Initializing environment and launching Streamlit web dashboard...
echo.

cd /d "%~dp0"

:: Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    color 0C
    echo [ERROR] Python was not found on your system PATH!
    echo Please install Python 3.10+ from https://www.python.org/
    echo and ensure "Add Python to PATH" is checked during installation.
    echo.
    pause
    exit /b 1
)

:: Activate virtual environment if present
if exist "venv\Scripts\activate.bat" (
    echo Activating local virtual environment: venv
    call "venv\Scripts\activate.bat"
)
if exist ".venv\Scripts\activate.bat" (
    echo Activating local virtual environment: .venv
    call ".venv\Scripts\activate.bat"
)

:: If running from root and app.py is in subdirectory, move in
if exist "SMS_SPAM_CLASSIFICATION\app.py" (
    if not exist "app.py" (
        cd SMS_SPAM_CLASSIFICATION
    )
)

echo.
echo Launching SMS Shield on http://localhost:8501 ...
echo Press Ctrl+C in this terminal window to stop the server.
echo.

python -m streamlit run app.py --server.showEmailPrompt false --browser.gatherUsageStats false

if errorlevel 1 (
    echo.
    echo [NOTICE] Streamlit server stopped.
    pause
)
