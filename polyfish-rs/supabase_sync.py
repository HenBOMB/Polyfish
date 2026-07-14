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
    key = os.environ.get("SUPABASE_KEY")
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
            checkpoints = sorted(glob.glob("checkpoints/model_checkpoint_iter*.safetensors"))
            if checkpoints:
                latest_cp = checkpoints[-1]
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

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python supabase_sync.py [upload|download] [file_path]")
        sys.exit(1)
    
    action = sys.argv[1]
    file_path = sys.argv[2] if len(sys.argv) > 2 else "model.safetensors"
    
    if action == "upload":
        upload(file_path)
    elif action == "download":
        download(file_path)
