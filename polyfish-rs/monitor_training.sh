#!/bin/bash
pid=756012
log_file="session.log"
last_size=$(stat -c%s "$log_file")

echo "Starting background monitoring for PID $pid..."

while kill -0 $pid 2>/dev/null; do
  sleep 1800
  new_size=$(stat -c%s "$log_file")
  if [ "$new_size" -eq "$last_size" ]; then
    echo "ERROR: $log_file has not grown in 30 minutes. Process $pid may be hung."
    exit 1
  fi
  last_size=$new_size
done

echo "Process $pid is no longer running. It may have finished or crashed."
