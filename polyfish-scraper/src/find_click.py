#!/usr/bin/env python3
"""
find_and_click.py

Usage:
  python3 find_and_click.py --pid 1234 --template ./button.png --threshold 0.8 --click

What it does:
 - Verifies PID exists.
 - Attempts to locate a window owned by that PID using `wmctrl -lpG`.
 - If found, screenshots that window area; otherwise screenshots entire screen.
 - Runs OpenCV template matching to find best match above threshold.
 - Optionally moves mouse and clicks the center of the matched region (using pyautogui).
"""

import argparse
import subprocess
import sys
import shutil
import time
from typing import Optional, Tuple, List

import psutil
import numpy as np
import cv2
from PIL import Image
import pyautogui

# ---------- Utilities ----------

def check_pid(pid: int) -> bool:
    try:
        p = psutil.Process(pid)
        return p.is_running()
    except Exception:
        return False

def run_cmd(cmd: List[str]) -> Tuple[int, str]:
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True)
        return r.returncode, r.stdout
    except Exception as e:
        return 1, f'ERR: {e}'

def read_pid_from_script(script_path="./read-pid.sh") -> Optional[int]:
    if not shutil.which(script_path):
        # allow executing local './read-pid.sh' even if not in PATH
        pass
    try:
        proc = subprocess.run([script_path], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True, check=False)
        raw = proc.stdout.strip()
        try:
            pid = int(raw)
        except ValueError:
            return None
        return pid
    except Exception:
        return None

def find_window_by_pid_wmctrl(pid: int) -> Optional[Tuple[int,int,int,int]]:
    """
    Use wmctrl -lpG to find a window with given PID.
    Returns (x, y, width, height) of the first matching window, or None.
    Requires wmctrl installed and X11.
    """
    if shutil.which("wmctrl") is None:
        return None
    code, out = run_cmd(["wmctrl", "-lpG"])
    if code != 0:
        return None
    # lines contain: 0x03600007  0 1234 host 0 0 800 600 window-title
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        try:
            win_pid = int(parts[2])
            if win_pid == pid:
                # fields after pid: host, x, y, w, h
                # depending on wmctrl, indexes may shift; using typical layout:
                # [0]=winid [1]=desktop [2]=pid [3]=host [4]=x [5]=y [6]=w [7]=h ...
                x = int(parts[4])
                y = int(parts[5])
                w = int(parts[6])
                h = int(parts[7])
                return (x, y, w, h)
        except Exception:
            continue
    return None

def screenshot_region(bbox: Optional[Tuple[int,int,int,int]] = None) -> Image.Image:
    """
    bbox: (left, top, width, height) or None -> full screen
    Returns a PIL Image.
    """
    if bbox is None:
        img = pyautogui.screenshot()
        return img
    left, top, w, h = bbox
    # pyautogui expects box=(left, top, width, height) via region param
    img = pyautogui.screenshot(region=(left, top, w, h))
    return img

def find_template_in_image(
    haystack_pil: Image.Image,
    needle_path: str,
    method=cv2.TM_CCOEFF_NORMED
) -> Optional[Tuple[int,int,float,Tuple[int,int]]]:
    """
    Returns (match_left, match_top, score, (w,h)) in haystack coords (if haystack is a full-screen or window crop).
    """
    haystack = cv2.cvtColor(np.array(haystack_pil), cv2.COLOR_RGB2BGR)
    needle = cv2.imread(needle_path, cv2.IMREAD_COLOR)
    if needle is None:
        raise FileNotFoundError(f"Template file not found or unreadable: {needle_path}")
    h,w,_ = needle.shape
    res = cv2.matchTemplate(haystack, needle, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        best_loc = min_loc
        best_score = min_val
    else:
        best_loc = max_loc
        best_score = max_val
    return (best_loc[0], best_loc[1], float(best_score), (w,h))

# ---------- Main flow ----------

def main():
    parser = argparse.ArgumentParser(description="Find template inside app window (PID from read-pid.sh) and click it.")
    parser.add_argument("--template", type=str, required=True, help="Path to template image")
    parser.add_argument("--threshold", type=float, default=0.8, help="Matching threshold (0..1)")
    parser.add_argument("--click", action="store_true", help="Click the found match")
    parser.add_argument("--pause-before-click", type=float, default=0.15, help="Delay before clicking")
    parser.add_argument("--pid-override", type=int, help="Manually override PID from script")
    parser.add_argument("--script", type=str, default="./get-pid.sh", help="PID script path")
    args = parser.parse_args()

    # --- PID acquisition phase ---
    if args.pid_override is not None:
        pid = args.pid_override
        print(f"[+] Using manual PID override: {pid}")
    else:
        print(f"[+] Running PID reader script: {args.script}")
        pid = read_pid_from_script(args.script)
        if pid is None:
            print("[-] PID script did not return a valid number.")
            sys.exit(2)
        if pid == -1:
            print("[-] PID script returned -1 → target application not running.")
            sys.exit(3)
        print(f"[+] PID script reports target PID = {pid}")

    # Confirm PID exists
    if not check_pid(pid):
        print(f"[-] PID {pid} is not running or inaccessible.")
        sys.exit(4)
    print("[+] PID is alive.")

    # --- Window lookup ---
    geom = find_window_by_pid_wmctrl(pid)
    if geom:
        print(f"[+] Window geometry found: {geom}")
    else:
        print("[*] No window found via wmctrl; falling back to full-screen screenshot.")
        geom = None

    # --- Screenshot ---
    try:
        screenshot = screenshot_region(geom)
    except Exception as e:
        print(f"[-] Screenshot failed: {e}")
        sys.exit(5)

    # --- Template matching ---
    print("[+] Running template match...")
    try:
        left, top, score, (w, h) = find_template_in_image(screenshot, args.template)
    except Exception as e:
        print(f"[-] Matching error: {e}")
        sys.exit(6)

    print(f"[+] Best match at ({left},{top}), score={score:.4f}")

    # Convert to screen coords
    if geom:
        screen_x = geom[0] + left + w//2
        screen_y = geom[1] + top + h//2
    else:
        screen_x = left + w//2
        screen_y = top + h//2

    if score < args.threshold:
        print(f"[-] Score below threshold ({args.threshold}). No click.")
        sys.exit(7)

    print(f"[+] Match above threshold. Target = ({screen_x},{screen_y})")

    if args.click:
        print(f"[+] Clicking in {args.pause_before_click} seconds...")
        time.sleep(args.pause_before_click)
        try:
            pyautogui.moveTo(screen_x, screen_y, duration=0.2)
            pyautogui.click()
            print("[+] Click complete.")
        except Exception as e:
            print("[-] click failed:", e)
    else:
        print("[*] --click not passed; not clicking.")

    sys.exit(0)


if __name__ == "__main__":
    main()
