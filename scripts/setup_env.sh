#!/usr/bin/env bash
set -euo pipefail

cd /mnt/d/URA_challenge
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
