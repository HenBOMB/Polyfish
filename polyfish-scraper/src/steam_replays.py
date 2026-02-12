import os
import time, subprocess, json
from typing import Optional
from typing_extensions import Literal
from collections.abc import Mapping, Sequence, Set
from math import isnan
from typing import Any
from util import make_driver, ydo, save_and_merge

DIR_DATA = "src/scraper/data"
DIR_ROOT = f"{DIR_DATA}/training-data/"
DIR_URI_REPLAYS = f"{DIR_DATA}/replays_"

REPLAY_TIMEOUT = 5
LOOKUP_TIMEOUT = 15
_MISSING = object()

os.makedirs(DIR_DATA, exist_ok=True)
os.makedirs(DIR_ROOT, exist_ok=True)

def _find_pid(process_name="Polytopia.exe") -> Optional[int]:
    """Find the PID of the game process (cached for the session)."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", process_name],
            capture_output=True, text=True, timeout=5
        )
        pids = result.stdout.strip().split('\n')
        if pids and pids[0]:
            return int(pids[-1])  # Latest PID
    except Exception:
        pass
    return None

class DaemonScanner:
    """Persistent polyfish-reader process for fast state polling."""
    
    def __init__(self):
        self._process: Optional[subprocess.Popen] = None
        self._pid: Optional[int] = None
    
    def start(self) -> bool:
        """Start the daemon. Returns True if successful."""
        self._pid = _find_pid()
        if not self._pid:
            print("[scanner] Game process not found")
            return False
        
        try:
            self._process = subprocess.Popen(
                ["sudo", "polyfish-reader/polyfish-reader", str(self._pid), "-y", "-d"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0  # Unbuffered
            )
            # Wait for "ready" message on stderr
            import select
            ready = select.select([self._process.stderr], [], [], 5.0)
            if ready[0]:
                msg = self._process.stderr.readline().decode('utf-8', errors='replace').strip()
                if 'ready' in msg:
                    print(f"[scanner] Daemon started (pid={self._pid})")
                    return True
                else:
                    print(f"[scanner] Unexpected startup message: {msg}")
            else:
                print("[scanner] Daemon did not report ready in time")
        except Exception as e:
            print(f"[scanner] Failed to start daemon: {e}")
        
        self.stop()
        return False
    
    def scan(self) -> Optional[str]:
        """Request a state scan. Returns JSON string or None."""
        if not self._process or self._process.poll() is not None:
            return None
        
        try:
            import select
            self._process.stdin.write(b"scan\n")
            self._process.stdin.flush()
            # Wait up to 2s for a response to avoid blocking forever
            ready = select.select([self._process.stdout], [], [], 2.0)
            if not ready[0]:
                return None
            line = self._process.stdout.readline()
            if not line:
                return None
            result = line.decode('utf-8').strip()
            return result if result else None
        except (BrokenPipeError, OSError):
            return None
    
    def stop(self):
        """Stop the daemon process."""
        if self._process:
            try:
                self._process.stdin.write(b"quit\n")
                self._process.stdin.flush()
                self._process.wait(timeout=2)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None
            print("[scanner] Daemon stopped")
    
    @property
    def alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

# Global scanner instance
_scanner = DaemonScanner()

def scan() -> Optional[str]:
    """Scan game state. Uses daemon mode if available, falls back to subprocess."""
    if _scanner.alive:
        return _scanner.scan()
    # Fallback to old method
    process = subprocess.Popen(["./scan.sh", "-y"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()
    if error:
        return None
    return output.decode('utf-8').strip()

def drag_replay(start_x, start_y, end_x, end_y, hold_time=0.2):
    ydo(["mousemove", str(start_x), str(start_y)])
    time.sleep(0.05)

    ydo(["mousedown", "1"])
    time.sleep(0.05)

    ydo(["mousemove", str(end_x), str(end_y)])
    time.sleep(hold_time)

    ydo(["mouseup", "1"])

def click(x, y):
    ydo(["mousemove", str(x), str(y)])
    ydo(["click", "1"])
    
def _repr_val(v):
    """Short, readable representation for printing differences."""
    try:
        return repr(v)
    except Exception:
        return f"<unrepr {type(v).__name__}>"

def compute_delta(a: Any, b: Any, path: str = "") -> dict:
    """
    Recursively compare a and b, returning a dictionary of {path: new_value}.
    Only includes keys that have changed.
    """
    diffs = {}
    
    # If types are different or one is missing, it's a leaf change
    if type(a) != type(b):
        return {path: b}

    # Mapping (dict-like)
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        keys = set(a.keys()) | set(b.keys())
        for k in keys:
            va = a.get(k, _MISSING)
            vb = b.get(k, _MISSING)
            new_path = f"{path}.{k}" if path else str(k)
            if va is _MISSING:
                diffs[new_path] = vb
            elif vb is _MISSING:
                diffs[new_path] = None # Or a specific delete marker
            else:
                diffs.update(compute_delta(va, vb, new_path))
        return diffs

    # Sequence (list/tuple)
    if isinstance(a, Sequence) and isinstance(b, Sequence) and not isinstance(a, (str, bytes)):
        len_a, len_b = len(a), len(b)
        min_len = min(len_a, len_b)
        for i in range(min_len):
            diffs.update(compute_delta(a[i], b[i], f"{path}[{i}]"))
        if len_a != len_b:
            if len_a < len_b: # Added items
                for i in range(min_len, len_b):
                    diffs[f"{path}[{i}]"] = b[i]
            else: # Removed items
                diffs[f"{path}.len"] = len_b # Signal truncation
        return diffs

    # Fallback/Leaf: compare directly
    if a != b:
        return {path: b}

    return diffs

def print_diffs(diffs):
    if not diffs:
        print("No differences found.")
        return
    for p, va, vb in diffs:
        print(f"Path: {p}")
        print(f"  - A: {_repr_val(va)}")
        print(f"  - B: {_repr_val(vb)}")
        print("---")

# Outcomes
#  1 = win
# -1 = loss
#  0 = draw
#  2 = not yet defined
def save_training_data(state: dict, old_state: Optional[dict] = None, winner_id: Optional[str] = None, file_id: Optional[str] = None):
    game_id = file_id if file_id else state['settings']['gameId']
    path = DIR_ROOT + f"{game_id}.csv"
    
    copy = json.loads(json.dumps(state))
    # Keep essential settings (turn, currentPlayerTurnId, mode, size)
    # but strip noisy/internal fields that bloat deltas
    for k in ['_recentMoves', '_pendingRewards', '_fow', '_areYouSure', '_gameOver',
              '_lastPlayerTurnId', 'gameName', 'version', 'gameId', 'seed',
              '_maxTribeCount', 'winByCapital', 'winByExtermination']:
        copy.get('settings', {}).pop(k, None)

    # Outcome logic
    # 1  = Won
    # -1 = Lost
    # 0  = Incomplete/Draw
    outcome = 0 
    if winner_id:
        current_player_id = str(state['settings']['currentPlayerTurnId'])
        outcome = 1 if current_player_id == str(winner_id) else -1

    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write("outcome,type,state\n")
            f.write(f"{outcome},base,{json.dumps(copy)}\n")
        return

    if old_state:
        old_copy = json.loads(json.dumps(old_state))
        for k in ['_recentMoves', '_pendingRewards', '_fow', '_areYouSure', '_gameOver',
                  '_lastPlayerTurnId', 'gameName', 'version', 'gameId', 'seed',
                  '_maxTribeCount', 'winByCapital', 'winByExtermination']:
            old_copy.get('settings', {}).pop(k, None)
        
        delta = compute_delta(old_copy, copy)
        if not delta: return

        with open(path, "a") as f:
            f.write(f"{outcome},delta,{json.dumps(delta)}\n")
    else:
        with open(path, "a") as f:
            f.write(f"{outcome},base,{json.dumps(copy)}\n")

def get_winner(state: Optional[dict]) -> Optional[str]:
    if not state:
        return None
    
    tribes_count = len(state['tribes'])
    defeat_count = 0
    standing_tribes = list(state['tribes'].keys())

    for id in state['tribes']:
        tribe = state['tribes'][id]
        if tribe['killedTurn'] > -1:
            defeat_count += 1
            standing_tribes.remove(id)
        elif tribe['resignedTurn'] > -1:
            defeat_count += 1
            standing_tribes.remove(id)

    if len(standing_tribes) == 1 or defeat_count == tribes_count - 1:
        return standing_tribes[0]
    else:
        return None
    
def magic(target: Literal['discord', 'polysseum', 'reddit', 'yt', '*']):
    dir_notfound = DIR_URI_REPLAYS + target + ".404.txt"
    dir_failed = DIR_URI_REPLAYS + target + ".failed.txt"
    dir_done = DIR_URI_REPLAYS + target + ".done.txt"

    fatal_urls = open(dir_notfound, "r").read().split('\n') if os.path.exists(dir_notfound) else []
    failed_urls = open(dir_failed, "r").read().split('\n') if os.path.exists(dir_failed) else []
    completed_urls = open(dir_done, "r").read().split('\n') if os.path.exists(dir_done) else []
    driver = make_driver()
    filename = DIR_URI_REPLAYS + target + ".txt"
    replay_urls = open(filename, "r").read().split('\n')
    replay_urls.extend(failed_urls)

    for i, url in enumerate(replay_urls):
        if url in completed_urls or url in fatal_urls:
            continue

        driver.get(url)
        time.sleep(3)

        # runs the steam game and opens the replay
        # if not already in a replay
        if not driver.execute_script("""
        try {
            document.querySelector('a.button[href*=\"steam://run/\"]').click();
            return true;
        }
        catch(e) {
            return false;
        }
        """):
            print('failed to load replay from url')
            save_and_merge(dir_failed, [url])
            save_and_merge(dir_failed, [url])
            continue
        
        # Extract UUID from URL
        # Format 1: https://share.polytopia.io/g/UUID
        # Format 2: steam://run/874390//opengame?id=UUID
        file_id = None
        if "id=" in url:
             try:
                 file_id = url.split("id=")[1].split("&")[0]
             except: pass
        elif "/g/" in url:
             try:
                 file_id = url.split("/g/")[1].split("/")[0]
             except: pass
        else:
             file_id = url.split("/")[-1]

        failed = False
        timeout = time.time()

        # Start daemon immediately so it's ready for fast polling
        _scanner.start()

        game_states = []
        old_state = None
        last_change_time = None
        replay_detected = False

        while True:
            time.sleep(0.001)

            # Phase 1: Waiting for replay to load
            if not replay_detected:
                if time.time() - timeout > LOOKUP_TIMEOUT:
                    print('lobby not found')
                    save_and_merge(dir_notfound, [url])
                    time.sleep(3)
                    failed = True
                    break

                raw = scan()
                if not raw:
                    continue

                print('replay loaded')
                replay_detected = True
                timeout = time.time()
                last_change_time = time.time()

                # Process the first scan result immediately
                state = json.loads(raw)
                cur_turn = state['settings']['currentPlayerTurnId']
                if cur_turn != 0:
                    old_state = state
                    game_states.append((state, None))
                continue

            # Phase 2: Collecting replay states
            if time.time() - timeout > REPLAY_TIMEOUT or get_winner(old_state):
                winner = get_winner(old_state)
                print(f'game ended (Winner: {winner})')
                
                # Now that the game is over, save all buffered states with the outcome
                print(f"Finalizing {len(game_states)} states...")
                for s, old_s in game_states:
                    save_training_data(s, old_s, winner, file_id=file_id)
                break

            raw = scan()
            if not raw:
                # Replay likely crashed or exited
                winner = get_winner(old_state)
                for s, old_s in game_states:
                    save_training_data(s, old_s, winner, file_id=file_id)
                break

            state = json.loads(raw)
            cur_turn = state['settings']['currentPlayerTurnId']

            if cur_turn == 0:
                continue

            if compute_delta(old_state, state):
                timeout = time.time()
                last_change_time = time.time()
                game_states.append((state, old_state))

            # Detect stale state (replay finished but didn't exit to menu)
            if time.time() - last_change_time > 3.0:
                 print("No state changes for 3s, assuming game over.")
                 winner = get_winner(old_state)
                 for s, old_s in game_states:
                     save_training_data(s, old_s, winner, file_id=file_id)
                 break

            old_state = state
        
        if failed:
            _scanner.stop()
            print('failed')
            continue

        print(f"+ {i+1}/{len(replay_urls)}")
        save_and_merge(dir_done, [url])
        
        # Stop daemon before menu navigation
        _scanner.stop()
        
        # finish turn
        click(825, 710)

        time.sleep(3)
        
        # on finish turn it will open the podeum
        # keep checking until we are in the menu
        while True:
            # done
            click(700, 500)

            time.sleep(3.0)
            
            if not scan():
                print('at menu')
                break
    
    driver.close()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and (sys.argv[1].startswith("http") or sys.argv[1].startswith("steam://")):
        # Test mode: run a single URL
        url = sys.argv[1]
        print(f"Running test on: {url}")
        
        file_id = None
        if "id=" in url:
             file_id = url.split("id=")[1].split("&")[0]
        elif "/g/" in url:
             file_id = url.split("/g/")[1].split("/")[0]
        else:
             file_id = url.split("/")[-1]
        print(f"Using File ID: {file_id}")

        if url.startswith("http"):
            driver = make_driver()
            driver.get(url)
            time.sleep(3)
            # Trigger Steam
            success = driver.execute_script("""
            try {
                document.querySelector('a.button[href*="steam://run/"]').click();
                return true;
            } catch(e) { return false; }
            """)
            if not success:
                print("Failed to trigger Steam button")
                driver.quit()
                sys.exit(1)
            print("Steam triggered via browser.")
            driver.quit()
        else:
            # Trigger via shell
            print("Triggering Steam directly via xdg-open...")
            subprocess.run(["xdg-open", url])
            
        print("Waiting for game to load replay...")
        
        old_state = None
        timeout = time.time()
        
        # Start daemon immediately for fast polling
        _scanner.start()

        # Unified detection + collection loop
        game_states = []
        last_change_time = None
        replay_detected = False
        
        while True:
            time.sleep(0.001)

            # Phase 1: Waiting for replay to load
            if not replay_detected:
                if time.time() - timeout > LOOKUP_TIMEOUT:
                    print("Timeout: Game didn't enter replay state.")
                    _scanner.stop()
                    sys.exit(1)

                raw = scan()
                if not raw:
                    continue

                print("Replay detected in game!")
                replay_detected = True
                timeout = time.time()
                last_change_time = time.time()

                # Process the first scan result immediately
                state = json.loads(raw)
                cur_turn = state['settings']['currentPlayerTurnId']
                if cur_turn != 0:
                    old_state = state
                    game_states.append((state, None))
                continue

            # Phase 2: Collecting replay states
            if time.time() - timeout > REPLAY_TIMEOUT or get_winner(old_state):
                winner = get_winner(old_state)
                print(f"Game ended or timeout reached (Winner: {winner})")
                for s, old_s in game_states:
                    save_training_data(s, old_s, winner, file_id=file_id)
                break
            
            raw = scan()
            if not raw: break
            
            state = json.loads(raw)
            cur_turn = state['settings']['currentPlayerTurnId']
            if cur_turn == 0: continue
            
            if compute_delta(old_state, state):
                timeout = time.time()
                last_change_time = time.time()
                game_states.append((state, old_state))
                print(f"Turn {state['settings']['turn']} captured")
            
            # Detect stale state
            if time.time() - last_change_time > 5.0:
                 print("No state changes for 5s, assuming game over.")
                 winner = get_winner(old_state)
                 for s, old_s in game_states:
                     save_training_data(s, old_s, winner, file_id=file_id)
                 break
            
            old_state = state
            
        _scanner.stop()
        print("Test complete.")
    else:
        magic('polysseum')
