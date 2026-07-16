#!/usr/bin/env python3
import os
import sys
import time
import subprocess
import tarfile
from datetime import datetime, timezone

# Add current dir to path to import supabase_sync
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    import supabase_sync
except ImportError:
    print("Could not import supabase_sync. Ensure this script is run from the workspace directory.")
    sys.exit(1)

POD_ID = "pbusxk8i78fao1"
TARGET_ITER = 186
MAX_HOURS = 8.0
MAX_SECONDS = MAX_HOURS * 3600
CHECK_INTERVAL_SECONDS = 300  # 5 minutes

def get_latest_iteration() -> int:
    """Downloads pod_state.tar.gz from Supabase, extracts training_log.csv, and reads the latest iteration."""
    temp_tar = "temp_pod_state.tar.gz"
    temp_csv = "temp_training_log.csv"
    
    # Clean up old temp files
    for temp_file in [temp_tar, temp_csv]:
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception:
                pass

    try:
        # Download state
        supabase = supabase_sync.get_client()
        data = supabase.storage.from_('models').download('pod_state.tar.gz')
        with open(temp_tar, 'wb') as f:
            f.write(data)
            
        # Extract training_log.csv
        with tarfile.open(temp_tar, "r:gz") as tar:
            tar.extract("training_log.csv", path=".")
            os.rename("training_log.csv", temp_csv)
            
        # Read iteration from the last row of CSV
        if not os.path.exists(temp_csv):
            print("extracted training_log.csv not found.")
            return -1
            
        import csv
        with open(temp_csv, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if not rows:
                return -1
            # Find rows matching active run_id if needed, but last row of log has the latest iteration
            last_row = rows[-1]
            return int(last_row['iteration'])
            
    except Exception as e:
        print(f"Error checking Supabase state: {e}")
        return -1
    finally:
        # Clean up temp files
        for temp_file in [temp_tar, temp_csv]:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception:
                    pass

def stop_pod():
    print(f"[{datetime.now().isoformat()}] Stopping pod {POD_ID}...")
    try:
        res = subprocess.run(
            ["/usr/local/bin/runpodctl", "pod", "stop", POD_ID],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"Successfully stopped pod. Output:\n{res.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to stop pod: {e}\nStderr:\n{e.stderr}")

def monitor():
    start_time = time.time()
    print(f"[{datetime.now().isoformat()}] Starting monitor for pod {POD_ID}.")
    print(f"Target: Iteration {TARGET_ITER} or elapsed time > {MAX_HOURS} hours.")
    
    while True:
        elapsed = time.time() - start_time
        if elapsed >= MAX_SECONDS:
            print(f"[{datetime.now().isoformat()}] Time limit of {MAX_HOURS} hours reached. Stopping pod.")
            stop_pod()
            break
            
        latest_iter = get_latest_iteration()
        if latest_iter != -1:
            print(f"[{datetime.now().isoformat()}] Latest iteration completed: {latest_iter} | Target: {TARGET_ITER} | Elapsed: {elapsed/3600:.2f}/{MAX_HOURS} hours")
            
            if latest_iter >= TARGET_ITER:
                print(f"[{datetime.now().isoformat()}] Target iteration {TARGET_ITER} reached (current: {latest_iter}). Stopping pod.")
                stop_pod()
                break
        else:
            print(f"[{datetime.now().isoformat()}] Warning: Could not retrieve latest iteration. Will retry.")
            
        time.sleep(CHECK_INTERVAL_SECONDS)

if __name__ == "__main__":
    monitor()
