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
        
        os.write(fd, b"curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y\n")
        time.sleep(15)
        
        os.write(fd, b"source $HOME/.cargo/env\n")
        time.sleep(1)
        
        os.write(fd, b"(cd /app && cargo build --release --bin arena && SEEDS=2 MCTS=64 ./rate_checkpoints.sh > bench_output.log 2>&1) &\n")
        time.sleep(2)
        
        os.write(fd, b"exit\n")
        time.sleep(2)
        
        try:
            os.waitpid(pid, os.WNOHANG)
        except OSError:
            pass

if __name__ == "__main__":
    transfer()
