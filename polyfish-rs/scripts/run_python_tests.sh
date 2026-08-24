#!/bin/bash
# Python-side test suite: ladder.py (gauge statistics) and train.py (the
# trainer the loop actually runs). Audit T3 recorded that neither had any test
# infrastructure.
#
# Stdlib unittest, no pytest: requirements.txt pins the training env and adding
# a test-only dependency to it would mean every training box installs it.
#
# Prefers .venv (where torch lives) and falls back to bare python3, in which
# case the torch-dependent cases skip and the Rust<->Python width parity cases
# still run. Both modes are expected to exit 0.
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -x .venv/bin/python ]; then
    PY=.venv/bin/python
else
    PY=python3
    echo "note: no .venv, running on bare python3 — torch-dependent cases will skip"
fi

exec "$PY" -m unittest discover -s tests -p "test_*.py" "$@"
