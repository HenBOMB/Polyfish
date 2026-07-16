import pty
import os
import time
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
        output_buffer = []
        
        def drain():
            while True:
                r, _, _ = select.select([fd], [], [], 0.5)
                if r:
                    try:
                        data = os.read(fd, 65536)
                        if data:
                            output_buffer.append(data)
                    except OSError:
                        break
        
        t = threading.Thread(target=drain, daemon=True)
        t.start()
        
        time.sleep(8)
        
        os.write(fd, b"apt-get update && apt-get install -y cargo\n")
        time.sleep(40)
        
        os.write(fd, b"cd /app && rm -f /app/target/release/arena && cargo build --release --bin arena > cargo_build.log 2>&1\n")
        time.sleep(60) # wait a bit for build
        
        os.write(fd, b"cat /app/cargo_build.log\n")
        time.sleep(2)
        
        os.write(fd, b"SEEDS=2 MCTS=64 ./rate_checkpoints.sh > bench_output.log 2>&1 &\n")
        time.sleep(2)
        
        os.write(fd, b"exit\n")
        time.sleep(2)
        
        try:
            os.waitpid(pid, os.WNOHANG)
        except OSError:
            pass
            
        with open("apt_cargo_build.log", "wb") as f:
            f.write(b"".join(output_buffer))

if __name__ == "__main__":
    transfer()
