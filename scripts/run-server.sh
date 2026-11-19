#!/bin/bash

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)" || exit 1
cd -- "$script_dir/../polyfish-rs" || exit 1

pids=$(lsof -t -i:3000)
if [[ -n "$pids" ]]; then
    while IFS= read -r pid; do
        kill "$pid" || true
    done <<< "$pids"
fi
exec cargo run --bin polyfish "$@"
