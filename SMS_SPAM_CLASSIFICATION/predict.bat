@echo off
title SMS SHIELD - Interactive SMS Predictor
color 0B
cls

cd /d "%~dp0"

:: Verify Python is installed and accessible
python --version >nul 2>&1
if errorlevel 1 (
    color 0C
    echo [ERROR] Python was not found on your system PATH!
    pause
    exit /b 1
)

:: Activate virtual environment if present
if exist "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
)
if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
)

:: Run interactive predictor
python predict.py %*

if errorlevel 1 (
    echo.
    echo [NOTICE] Session ended.
    pause
)
