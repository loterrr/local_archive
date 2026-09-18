#!/usr/bin/env bash
set -euo pipefail
python -m compileall -q .
pytest -q
python app/server.py
