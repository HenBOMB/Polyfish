import pty
import os
import time
import base64
import select
import threading

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
        def drain():
            while True:
                r, _, _ = select.select([fd], [], [], 0.5)
                if r:
                    try:
                        os.read(fd, 65536)
                    except OSError:
                        break
        
        t = threading.Thread(target=drain, daemon=True)
        t.start()
        
        time.sleep(8)
        
        os.write(fd, b"stty -echo\n")
        time.sleep(1)
        
        os.write(fd, b"mkdir -p /app/target/release\n")
        time.sleep(1)
        
        with open("target/release/arena", "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode("utf-8")
        
        os.write(fd, b"cat << 'EOF_ARENA' > /app/target/release/arena.b64\n")
        time.sleep(1)
        
        chunk_size = 4096
        for i in range(0, len(b64), chunk_size):
            os.write(fd, b64[i:i+chunk_size].encode())
            time.sleep(0.01)
            
        os.write(fd, b"\nEOF_ARENA\n")
        time.sleep(2)
        
        os.write(fd, b"base64 -d /app/target/release/arena.b64 > /app/target/release/arena\n")
        time.sleep(2)
        
        os.write(fd, b"chmod +x /app/target/release/arena\n")
        time.sleep(1)
        
        os.write(fd, b"cd /app && SEEDS=2 MCTS=64 ./rate_checkpoints.sh > bench_output.log 2>&1 &\n")
        time.sleep(1)
        
        os.write(fd, b"exit\n")
        time.sleep(2)
        
        try:
            os.waitpid(pid, os.WNOHANG)
        except OSError:
            pass

if __name__ == "__main__":
    transfer()
