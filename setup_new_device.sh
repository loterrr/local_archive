#!/usr/bin/env bash
set -e

echo "============================================================"
echo "  LocalRAG: Automated Setup for macOS / Linux"
echo "============================================================"
echo ""

# 1. Check Python
if command -v python3 &>/dev/null; then
    PYTHON_CMD=python3
elif command -v python &>/dev/null; then
    PYTHON_CMD=python
else
    echo "[ERROR] Python 3 is not installed or not in PATH."
    echo "Please install Python 3.10 or 3.11."
    exit 1
fi

echo "[1/5] Detected Python: $($PYTHON_CMD --version)"

# 2. Create Virtual Environment
if [ ! -d ".venv" ]; then
    echo "[2/5] Creating virtual environment (.venv)..."
    $PYTHON_CMD -m venv .venv
else
    echo "[2/5] Virtual environment (.venv) already exists."
fi

# 3. Activate and Install Requirements
echo "[3/5] Installing dependencies into .venv..."
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 4. Create .env if missing
if [ ! -f ".env" ]; then
    echo "[4/5] Creating .env from .env.example..."
    cp .env.example .env
else
    echo "[4/5] .env file already exists."
fi

# 5. Check Ollama
echo "[5/5] Checking Ollama..."
if command -v ollama &>/dev/null; then
    echo "Ollama detected! Pulling phi3.5:latest..."
    ollama pull phi3.5:latest || true
else
    echo "[WARNING] Ollama is not installed or not running."
    echo "Please install Ollama from https://ollama.com"
fi

echo ""
echo "============================================================"
echo "  Setup Complete!"
echo "============================================================"
echo "To start the server, run:"
echo "  source .venv/bin/activate"
echo "  python app/server.py"
echo "Then open your browser at: http://localhost:3000"
echo "============================================================"
