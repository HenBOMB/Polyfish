#!/bin/bash
./run_training_loop.sh --resume -K 5 -f &
pid=$!

log_file="session.log"
# Wait a moment for session.log to be appended
sleep 5
last_size=$(stat -c%s "$log_file" 2>/dev/null || echo 0)
echo "Started training with PID $pid, monitoring for hangs..."

while kill -0 $pid 2>/dev/null; do
  sleep 1800
  new_size=$(stat -c%s "$log_file" 2>/dev/null || echo 0)
  if [ "$new_size" -eq "$last_size" ]; then
    echo "ERROR: $log_file has not grown in 30 minutes. Process $pid may be hung."
    kill -9 $pid
    pkill -P $pid
    exit 1
  fi
  last_size=$new_size
done

wait $pid
echo "Training finished with exit code $?"
