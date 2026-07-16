import os
import sys

try:
    from supabase import create_client, Client
except ImportError:
    print("Supabase package not installed. Run: pip install supabase")
    sys.exit(1)

def get_client() -> Client:
    from dotenv import load_dotenv
    load_dotenv()
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        print("Missing SUPABASE_URL or SUPABASE_KEY in .env. Skipping Supabase sync.")
        sys.exit(0)
    return create_client(url, key)

def upload(file_path: str, bucket: str = "models"):
    if not os.path.exists(file_path):
        print(f"File {file_path} not found. Skipping upload.")
        sys.exit(0)
    
    supabase = get_client()
    filename = os.path.basename(file_path)
    print(f"⬆️ Uploading {file_path} to Supabase bucket '{bucket}'...")
    
    try:
        with open(file_path, 'rb') as f:
            # We use upsert to overwrite the existing latest model
            supabase.storage.from_(bucket).upload(
                file=f,
                path=filename,
                file_options={"cacheControl": "3600", "upsert": "true"}
            )
        print("✅ Supabase upload complete.")
        
        # --- NEW: Also upload the latest checkpoint so we keep history ---
        if filename == "model.safetensors":
            import glob
            checkpoints = glob.glob("checkpoints/model_checkpoint_iter*.safetensors")
            if checkpoints:
                latest_cp = max(checkpoints, key=os.path.getmtime)
                cp_name = os.path.basename(latest_cp)
                print(f"⬆️ Backing up latest checkpoint: {cp_name} to Supabase...")
                with open(latest_cp, 'rb') as f_cp:
                    supabase.storage.from_(bucket).upload(
                        file=f_cp,
                        path=cp_name,
                        file_options={"cacheControl": "3600", "upsert": "true"}
                    )
                print(f"✅ Checkpoint {cp_name} backed up.")
                
    except Exception as e:
        print(f"⚠️ Failed to upload to Supabase: {e}")

def download(file_path: str, bucket: str = "models"):
    if os.path.exists(file_path):
        print(f"File {file_path} already exists locally. Skipping download.")
        sys.exit(0)
        
    supabase = get_client()
    filename = os.path.basename(file_path)
    print(f"⬇️ Attempting to download {filename} from Supabase bucket '{bucket}'...")
    
    try:
        data = supabase.storage.from_(bucket).download(filename)
        with open(file_path, 'wb') as f:
            f.write(data)
        print(f"✅ Download complete: {file_path}")
    except Exception as e:
        print(f"⚠️ Could not download {filename} from Supabase (may not exist yet).")

def download_checkpoint_iter(target_iter: int, bucket: str = "models") -> str | None:
    """Download the checkpoint for a specific iteration from Supabase and save as model.safetensors.
    
    Returns the matched checkpoint filename, or None if not found.
    """
    import re
    supabase = get_client()
    print(f"⬇️ Looking for checkpoint iter {target_iter} in Supabase bucket '{bucket}'...")
    
    try:
        files = supabase.storage.from_(bucket).list()
        # Find checkpoint(s) matching the exact iteration number
        pattern = re.compile(r"^model_checkpoint_iter(\d+)_.*\.safetensors$")
        matches = []
        for f in files:
            name = f.get("name", "")
            m = pattern.match(name)
            if m and int(m.group(1)) == target_iter:
                matches.append(name)
        
        if not matches:
            print(f"❌ No checkpoint found for iteration {target_iter} in Supabase.")
            return None
        
        # If multiple timestamps exist for the same iter, pick the latest (lexicographic sort on timestamp suffix)
        chosen = sorted(matches)[-1]
        print(f"⬇️ Downloading checkpoint: {chosen}...")
        data = supabase.storage.from_(bucket).download(chosen)
        with open("model.safetensors", 'wb') as out_f:
            out_f.write(data)
        print(f"✅ Checkpoint {chosen} downloaded and saved as model.safetensors")
        return chosen
    except Exception as e:
        print(f"⚠️ Failed to download checkpoint for iter {target_iter}: {e}")
        return None


def download_all_checkpoints(bucket: str = "models", min_iter: int = 0, matches_file: str = None):
    supabase = get_client()
    print(f"⬇️ Listing files in Supabase bucket '{bucket}'...")
    
    evaluated_content = ""
    if matches_file and os.path.exists(matches_file):
        with open(matches_file, "r") as mf:
            evaluated_content = mf.read()
            
    try:
        files = supabase.storage.from_(bucket).list()
        os.makedirs("checkpoints", exist_ok=True)
        for f in files:
            name = f.get("name")
            if name and name.startswith("model_checkpoint_iter") and name.endswith(".safetensors"):
                if min_iter > 0:
                    import re
                    m = re.search(r"iter(\d+)(?:_.*)?\.safetensors", name)
                    if m and int(m.group(1)) < min_iter:
                        continue
                        
                if evaluated_content:
                    base_name = name.replace(".safetensors", "")
                    if f'"{base_name}@' in evaluated_content:
                        print(f"⏭️ Skipping already evaluated checkpoint: {name}")
                        continue

                local_path = os.path.join("checkpoints", name)
                if not os.path.exists(local_path):
                    print(f"⬇️ Downloading missing checkpoint: {name}...")
                    data = supabase.storage.from_(bucket).download(name)
                    with open(local_path, 'wb') as out_f:
                        out_f.write(data)
        print("✅ Checkpoint sync complete.")
    except Exception as e:
        print(f"⚠️ Failed to sync checkpoints: {e}")

def backup_pod(bucket: str = "models"):
    import subprocess
    print("📦 Creating pod_state.tar.gz...")
    files_to_backup = [
        "training_log.csv",
        "config.json",
        ".last_train_metrics.json",
        ".last_self_play_metrics.json",
        ".anchor_state.json",
        "value_distribution.json",
        "moves_by_turn.json",
        "elo.log",
        "elo_ratings.json"
    ]
    # Only include files/dirs that exist
    existing_files = [f for f in files_to_backup if os.path.exists(f)]
    if not existing_files:
        print("⚠️ No state files found to backup.")
        return
        
    subprocess.run(["tar", "-czf", "pod_state.tar.gz"] + existing_files)
    upload("pod_state.tar.gz", bucket)

def restore_pod(bucket: str = "models"):
    import subprocess
    download("pod_state.tar.gz", bucket)
    if os.path.exists("pod_state.tar.gz"):
        print("📦 Extracting pod_state.tar.gz...")
        subprocess.run(["tar", "-xzf", "pod_state.tar.gz"])
        print("✅ Restore complete.")
    else:
        print("⚠️ No pod_state.tar.gz found to restore.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python supabase_sync.py [upload|download|download-checkpoint-iter|download-all-checkpoints] [file_path]")
        sys.exit(1)
    
    action = sys.argv[1]
    file_path = sys.argv[2] if len(sys.argv) > 2 else "model.safetensors"
    
    if action == "upload":
        upload(file_path)
    elif action == "download":
        download(file_path)
    elif action == "download-checkpoint-iter":
        if len(sys.argv) < 3:
            print("Usage: python supabase_sync.py download-checkpoint-iter <iteration_number>")
            sys.exit(1)
        target_iter = int(sys.argv[2])
        result = download_checkpoint_iter(target_iter)
        if result is None:
            sys.exit(1)
    elif action == "download-all-checkpoints":
        min_iter = 0
        matches_file = None
        if len(sys.argv) > 2:
            try:
                min_iter = int(sys.argv[2])
            except ValueError:
                pass
        if len(sys.argv) > 3:
            matches_file = sys.argv[3]
        download_all_checkpoints(min_iter=min_iter, matches_file=matches_file)
    elif action == "backup-pod":
        backup_pod()
    elif action == "restore-pod":
        restore_pod()
