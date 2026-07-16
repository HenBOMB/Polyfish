import pty
import os
import time
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
        
        os.write(fd, b"cat /app/bench_output.log\n")
        time.sleep(2)
        
        os.write(fd, b"echo 'SCRIPT_DONE'\n")
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
        
        with open("bench_output_local.log", "wb") as f:
            f.write(output)
            
        os.write(fd, b"exit\n")
        time.sleep(1)
        
        try:
            os.waitpid(pid, os.WNOHANG)
        except OSError:
            pass

if __name__ == "__main__":
    transfer()
