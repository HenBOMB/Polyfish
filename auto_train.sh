#!/bin/bash

# Configuration
# Replace this with the actual path to your training script
TRAIN_CMD="./run-server.sh" # Modify this to match your exact training command
CHECK_INTERVAL=1 # seconds

# Check for required tools
if ! command -v gdbus &> /dev/null; then
    echo "gdbus is required but not found."
    exit 1
fi

is_allowed_time() {
    local day=$(date +%w) # 0 = Sunday, 1-5 = Mon-Fri, 6 = Saturday
    local hour=$(date +%H)
    
    # Weekend: All day
    if [[ "$day" == "0" || "$day" == "6" ]]; then
        return 0
    fi
    
    # Weekday: 20:00 (8 PM) to 08:00 (8 AM)
    # Allowed hours: 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7
    if [[ "$hour" -ge 20 || "$hour" -lt 8 ]]; then
        return 0
    fi
    
    return 1
}

TRAIN_PID=""
IDLE_SECS=0
LAST_REPORT_DAY=""

echo "Starting auto-train monitor..."
echo "Training allowed: Weekdays 20:00-08:00, Weekends all day."
echo "Requires 60s of mouse inactivity."
echo "Checking every ${CHECK_INTERVAL} seconds..."

while true; do
    # AGY daily report at 8 PM (20:00)
    CURRENT_HOUR=$(date +%H)
    CURRENT_DAY=$(date +%Y-%m-%d)
    
    if [[ "$CURRENT_HOUR" == "20" && "$LAST_REPORT_DAY" != "$CURRENT_DAY" ]]; then
        echo "$(date): Triggering daily AGY training evaluation report..."
        # Running agy with the specific context needed for deep analysis (based on previous findings)
        agy "Perform a rigorous evaluation of the Polyfish/PolyZero training data. Please follow these exact steps:
1. Parse 'session.log' and 'training_log.csv' in the current directory to extract the latest training metrics.
2. Compare Policy Loss vs. Value Loss. If Value Loss is near zero (e.g., ~0.03) while Policy Loss remains high, explicitly flag that the 'value head has collapsed' and the model is producing near-random value gradients.
3. Calculate the timeout rate. Search 'session.log' for 'Decisive: true' vs 'Decisive: false'. If the vast majority of games are timing out instead of ending decisively, flag that the model is failing to learn long-term planning and closing strategies.
4. Read 'src/bin/self_play.rs' to identify the current curriculum phase (e.g., Tiny maps progressing from 10 to 30 max turns). Account for these phases when analyzing score trends, as map size and turn limit increases artificially inflate scores.
Write a comprehensive Markdown report summarizing the health of the training run based on these specific metrics." < /dev/null > "agy_report_$CURRENT_DAY.txt" 2>&1 &
        LAST_REPORT_DAY="$CURRENT_DAY"
    fi

    # Get idle time in milliseconds using gdbus (works on Wayland GNOME)
    IDLE_MS=$(gdbus call --session --dest org.gnome.Mutter.IdleMonitor --object-path /org/gnome/Mutter/IdleMonitor/Core --method org.gnome.Mutter.IdleMonitor.GetIdletime 2>/dev/null | grep -o "[0-9]\+" | tail -1)
    
    if [[ -n "$IDLE_MS" ]]; then
        IDLE_SECS=$((IDLE_MS / 1000))
    else
        IDLE_SECS=0
    fi
    
    # Check if there was activity (idle less than 60s)
    if [[ "$IDLE_SECS" -lt 60 ]]; then
        if [[ -n "$TRAIN_PID" ]]; then
            echo "$(date): Activity detected. Halting training script (PID: $TRAIN_PID)."
            # Kill the process abruptly
            kill -9 "$TRAIN_PID" 2>/dev/null
            
            # If your training script spawns child processes, uncomment the line below to kill them too:
            # pkill -9 -P "$TRAIN_PID" 
            
            TRAIN_PID=""
        fi
    else
        # Idle for >= 60s, check if current time is allowed
        if is_allowed_time; then
            if [[ -z "$TRAIN_PID" ]]; then
                echo "$(date): Conditions met (Idle for ${IDLE_SECS}s). Starting training script..."
                
                # Start the training script in the background
                $TRAIN_CMD &
                TRAIN_PID=$!
                
                echo "Training script started with PID: $TRAIN_PID"
            else
                # Check if the process is still running (maybe it crashed or finished)
                if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
                    echo "$(date): Training script exited unexpectedly. Restarting..."
                    $TRAIN_CMD &
                    TRAIN_PID=$!
                    echo "Training script restarted with PID: $TRAIN_PID"
                fi
            fi
        else
            # Outside of the allowed time window
            if [[ -n "$TRAIN_PID" ]]; then
                echo "$(date): Time window ended. Halting training script (PID: $TRAIN_PID)."
                kill -9 "$TRAIN_PID" 2>/dev/null
                TRAIN_PID=""
            fi
        fi
    fi
    
    sleep $CHECK_INTERVAL
done
