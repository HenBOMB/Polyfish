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
        
        os.write(fd, b"cd /workspace\n")
        time.sleep(1)
        
        # Tar files, ignore errors if some files don't exist
        os.write(fd, b"tar -czf backup.tar.gz *.json *.jsonl *.csv *.log .last* 2>/dev/null\n")
        time.sleep(5)
        
        script = """import base64
with open('backup.tar.gz', 'rb') as f:
    print(base64.b64encode(f.read()).decode('utf-8'))
"""
        os.write(fd, b"cat << 'EOF' > b64.py\n" + script.encode() + b"EOF\n")
        time.sleep(1)
        os.write(fd, b"python3 b64.py > backup.b64\n")
        time.sleep(5)
        
        # Now read base64 file
        os.write(fd, b"echo '---START_BASE64---'\n")
        time.sleep(0.5)
        os.write(fd, b"cat backup.b64\n")
        time.sleep(1)
        os.write(fd, b"echo '\n---END_BASE64---'\n")
        
        output = b""
        start = time.time()
        while True:
            # We need to wait enough time for cat to finish on large files
            r, _, _ = select.select([fd], [], [], 2)
            if r:
                try:
                    data = os.read(fd, 65536)
                    if not data:
                        break
                    output += data
                except OSError:
                    break
            else:
                if time.time() - start > 60:  # Timeout after 60s of no data
                    break
                
        output_str = output.decode('utf-8', errors='ignore')
        print(f"Total output received: {len(output_str)} characters")
        
        if '---START_BASE64---' in output_str and '---END_BASE64---' in output_str:
            b64_content = output_str.split('---START_BASE64---')[1].split('---END_BASE64---')[0]
            import re
            b64_content = re.sub(r'[^A-Za-z0-9+/=]', '', b64_content)
            
            try:
                with open("backup_from_pod.tar.gz", "wb") as f:
                    f.write(base64.b64decode(b64_content))
                print("Successfully downloaded backup_from_pod.tar.gz")
            except Exception as e:
                print(f"Error decoding base64: {e}")
        else:
            print("Could not find base64 markers in output.")
            print("Output tail:", output_str[-500:])
            
        os.write(fd, b"exit\n")
        time.sleep(1)
        
        try:
            os.waitpid(pid, os.WNOHANG)
        except OSError:
            pass

if __name__ == "__main__":
    transfer()
