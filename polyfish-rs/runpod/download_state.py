import pty
import os
import time
import select
import base64

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
        os.write(fd, b"export PS1=''\n")
        time.sleep(0.5)
        os.write(fd, b"export TERM=dumb\n")
        time.sleep(0.5)
        os.write(fd, b"stty -echo\n")
        time.sleep(1)
        os.write(fd, b"cd /app\n")
        time.sleep(1)
        
        script = """import base64
with open('pod_state.tar.gz', 'rb') as f:
    print(base64.b64encode(f.read()).decode('utf-8'))
"""
        os.write(fd, b"cat << 'EOF' > b64.py\n" + script.encode() + b"EOF\n")
        time.sleep(1)
        print("Encoding pod_state.tar.gz to base64 on remote...")
        os.write(fd, b"python3 b64.py > backup.b64\n")
        time.sleep(15) # Wait longer for 184MB encoding
        
        print("Transferring base64...")
        os.write(fd, b"echo '---START_BASE64---'\n")
        time.sleep(0.5)
        os.write(fd, b"cat backup.b64\n")
        time.sleep(1)
        os.write(fd, b"echo '\\n---END_BASE64---'\n")
        
        output = bytearray()
        start = time.time()
        while True:
            r, _, _ = select.select([fd], [], [], 2)
            if r:
                try:
                    data = os.read(fd, 1048576) # read in large chunks
                    if not data:
                        break
                    output.extend(data)
                    start = time.time()
                except OSError:
                    break
            else:
                if time.time() - start > 15:  # Timeout after 15s of no data
                    break
                
        output_str = output.decode('utf-8', errors='ignore')
        print(f"Total output received: {len(output_str)} characters")
        
        if '---START_BASE64---' in output_str and '---END_BASE64---' in output_str:
            b64_content = output_str.split('---START_BASE64---')[1].split('---END_BASE64---')[0]
            import re
            b64_content = re.sub(r'[^A-Za-z0-9+/=]', '', b64_content)
            try:
                with open("pod_state.tar.gz", "wb") as f:
                    f.write(base64.b64decode(b64_content))
                print("Successfully downloaded pod_state.tar.gz")
            except Exception as e:
                print(f"Error decoding base64: {e}")
        else:
            print("Could not find base64 markers in output.")
            
        os.write(fd, b"exit\n")
        time.sleep(1)
        try:
            os.waitpid(pid, os.WNOHANG)
        except OSError:
            pass

if __name__ == "__main__":
    transfer()
