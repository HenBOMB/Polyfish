#!/bin/bash

if ! command -v g++ &>/dev/null; then
    echo "Error: g++ is not installed!" 1>&2
    exit 1
fi

PROCESS_NAME="Polytopia.exe"

pid=$(ps -eo pid,lstart,cmd | grep "$PROCESS_NAME" | grep -v grep  | tail -n1 | awk '{print $1}')
# pid=$(ps -eo pid,lstart,cmd | grep "$PROCESS_NAME" | grep -v grep | sort -k2 | tail -n1 | awk '{print $1}')
# pid=$(pgrep -f "$PROCESS_NAME")

if [[ -z "$pid" ]]; then
    echo "Error: Process '$PROCESS_NAME' not found!" 1>&2
    exit 1
fi

# echo "Running polyfish-reader with PID: $pid"

# start_time=$(date +%s)

polyfish-reader/polyfish-reader "$pid" "$1"

# end_time=$(date +%s)
# elapsed_time=$((end_time - start_time))

# echo "Took: $elapsed_time seconds"