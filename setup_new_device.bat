@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo   LocalRAG: Automated Setup for Windows
echo ============================================================
echo.

:: 1. Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not found in your PATH!
    echo Please install Python 3.11 from python.org and check "Add python.exe to PATH".
    pause
    exit /b 1
)

echo [1/5] Checking Python installation...
python --version

:: 2. Create Virtual Environment
if not exist ".venv" (
    echo [2/5] Creating Python virtual environment (.venv)...
    python -m venv .venv
) else (
    echo [2/5] Virtual environment (.venv) already exists.
)

:: 3. Activate and Install Requirements
echo [3/5] Installing dependencies into .venv...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt

:: 4. Create .env if missing
if not exist ".env" (
    echo [4/5] Creating .env from .env.example...
    copy .env.example .env
) else (
    echo [4/5] .env file already exists.
)

:: 5. Check Ollama
echo [5/5] Checking Ollama status...
ollama --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Ollama is not installed or not in PATH.
    echo Please download and install Ollama from https://ollama.com
) else (
    echo Ollama detected! Checking model phi3.5:latest...
    ollama pull phi3.5:latest
)

echo.
echo ============================================================
echo   Setup Complete!
echo ============================================================
echo To start the server, run:
echo   .venv\Scripts\activate
echo   python app/server.py
echo Then open your browser at: http://localhost:3000
echo ============================================================
pause
