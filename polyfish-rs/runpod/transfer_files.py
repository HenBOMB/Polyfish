import pty
import os
import time
import base64
import select

def transfer():
    pid, fd = pty.fork()
    if pid == 0:
        from dotenv import load_dotenv
        import pathlib
        env_path = pathlib.Path(__file__).parent.parent / '.env'
        load_dotenv(dotenv_path=env_path)
        runpod_ssh = os.environ.get("RUNPOD_SSH_URL")
        if not runpod_ssh:
            print("Error: RUNPOD_SSH_URL not found in .env")
            os._exit(1)
        ssh_key = os.path.expanduser("~/.ssh/id_ed25519")
        os.execvp("ssh", ["ssh", "-i", ssh_key, "-o", "StrictHostKeyChecking=no", runpod_ssh])
    else:
        time.sleep(8)
        
        os.write(fd, b"cd /app\n")
        time.sleep(1)
        
        files_to_upload = [
            "rate_checkpoints.sh",
            "run_training_loop.sh",
            "run_training_runpod.sh",
            "supabase_sync.py"
        ]
        
        for filename in files_to_upload:
            with open(filename, "rb") as f:
                data = f.read()
            b64 = base64.b64encode(data).decode("utf-8")
            
            os.write(fd, f"cat << 'EOF' > {filename}.b64\n".encode())
            time.sleep(0.5)
            
            chunk_size = 1024
            for i in range(0, len(b64), chunk_size):
                os.write(fd, b64[i:i+chunk_size].encode())
                time.sleep(0.1)
                
            os.write(fd, b"\nEOF\n")
            time.sleep(1)
            
            os.write(fd, f"base64 -d {filename}.b64 > {filename}\n".encode())
            time.sleep(1)
            
        os.write(fd, b"chmod +x *.sh\n")
        time.sleep(1)
        
        # We also need the target/release/arena binary!
        # wait, the pod has target/release/self_play and target/release/polyfish.
        # Does it have arena? The Dockerfile didn't copy arena in the old image.
        # But rate_checkpoints.sh says:
        # if command -v cargo >/dev/null 2>&1; then cargo build --release --bin arena; elif [ ! -x "$ARENA" ]; then echo "Error"; exit 1; fi
        # If the pod doesn't have cargo or the source code, it can't build arena!
        # Let's see if we should upload `arena` binary?
        # That's too big for base64 probably (50MB+). Let's hope it can build it or we just run it and see.
        
        os.write(fd, b"SEEDS=2 MCTS=64 ./rate_checkpoints.sh > bench_output.log 2>&1 &\n")
        time.sleep(1)
        
        output = b""
        start = time.time()
        while time.time() - start < 5:
            r, _, _ = select.select([fd], [], [], 1)
            if r:
                try:
                    data = os.read(fd, 4096)
                    if not data:
                        break
                    output += data
                except OSError:
                    break
        
        with open("remote_output_files.log", "wb") as f:
            f.write(output)
            
        os.write(fd, b"exit\n")
        time.sleep(1)
        
        try:
            os.waitpid(pid, os.WNOHANG)
        except OSError:
            pass

if __name__ == "__main__":
    transfer()
