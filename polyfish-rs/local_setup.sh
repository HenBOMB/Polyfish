#!/bin/bash

if [[ "$(basename "$PWD")" != "polyfish-rs" ]]; then
    cd polyfish-rs || exit 1
fi

python3 -m venv .venv

.venv/bin/pip install -r requirements.txt