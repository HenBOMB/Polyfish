#!/bin/bash

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)" || exit 1
git -C "$script_dir/.." fetch
git -C "$script_dir/.." pull
