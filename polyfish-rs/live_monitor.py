#!/usr/bin/env python3
"""Live Terminal Dashboard for Remote RunPod & PolyZero Training.

Streams hardware telemetry, simulation velocity, time wasted, and money wasted
from Supabase directly to your local terminal.

Usage:
    cd polyfish-rs
    python3 live_monitor.py              # Continuous live dashboard (refreshes every 3s)
    python3 live_monitor.py --oneshot    # Print current status once and exit
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# Load Supabase
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    import supabase_sync
except ImportError:
    print("Error: Could not import supabase_sync. Run from polyfish-rs/ directory.")
    sys.exit(1)


def clear_screen():
    """Clear terminal screen and move cursor to top."""
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def render_bar(pct: float, width: int = 24, filled_char: str = "█", empty_char: str = "░") -> str:
    """Render a visual ASCII percentage bar."""
    clamped = max(0.0, min(100.0, pct))
    filled_len = int(round((clamped / 100.0) * width))
    empty_len = width - filled_len
    return f"[{filled_char * filled_len}{empty_char * empty_len}] {clamped:>5.1f}%"


def fetch_live_status(bucket: str = "models") -> Optional[Dict[str, Any]]:
    """Download the latest hardware_status.json from Supabase storage."""
    try:
        supabase = supabase_sync.get_client()
        data = supabase.storage.from_(bucket).download("hardware_status.json")
        return json.loads(data.decode("utf-8"))
    except Exception:
        return None


def format_duration(seconds: float) -> str:
    """Format seconds into HH:MM:SS or MM:SS."""
    secs = int(seconds)
    hours = secs // 3600
    minutes = (secs % 3600) // 60
    rem_secs = secs % 60
    if hours > 0:
        return f"{hours:02d}h {minutes:02d}m {rem_secs:02d}s"
    return f"{minutes:02d}m {rem_secs:02d}s"


def render_dashboard(data: Optional[Dict[str, Any]], error_msg: Optional[str] = None):
    """Render the full terminal UI dashboard."""
    clear_screen()
    now_str = datetime.now().strftime("%H:%M:%S")

    print("╔══════════════════════════════════════════════════════════════════════════════════╗")
    print(f"║ 🛰️  POLYZERO CLOUD POD TELEMETRY STREAM                     Local Time: {now_str:<9} ║")
    print("╚══════════════════════════════════════════════════════════════════════════════════╝")

    if not data:
        print("\n  🟡 Waiting for telemetry stream from pod...")
        if error_msg:
            print(f"  Note: {error_msg}")
        print("\n  Ensure your RunPod instance has started and SUPABASE credentials are set.")
        print("  Press Ctrl+C to exit.\n")
        return

    # Parse components
    updated_at = data.get("updated_at", "Unknown")
    iteration = data.get("latest_iteration", 0)
    duration = data.get("duration_sec", 0.0)
    moves_sec = data.get("moves_per_sec", 0.0)
    games_count = data.get("games_count", 0)

    gpu = data.get("gpu", {})
    cpu = data.get("cpu", {})
    fin = data.get("financials", {})
    diag = data.get("diagnosis", {})

    gpu_name = gpu.get("name", "Unknown GPU")
    gpu_util = gpu.get("util_avg_pct", 0.0)
    gpu_peak = gpu.get("util_max_pct", 0.0)
    vram_used = gpu.get("vram_used_mb", 0.0)
    vram_total = gpu.get("vram_total_mb", 0.0)
    vram_pct = gpu.get("vram_util_pct", 0.0)
    vram_waste = gpu.get("vram_waste_pct", 0.0)
    power = gpu.get("power_avg_w", 0.0)

    cpu_util = cpu.get("util_avg_pct", 0.0)
    cpu_peak = cpu.get("util_max_pct", 0.0)
    ram_used = cpu.get("ram_used_mb", 0.0)

    rate = fin.get("hourly_rate", 0.0)
    iter_cost = fin.get("iteration_cost", 0.0)
    iter_wasted_money = fin.get("iteration_money_wasted", 0.0)
    time_wasted = fin.get("time_wasted_sec", 0.0)
    waste_pct = fin.get("waste_pct", 0.0)
    cum_spend = fin.get("cumulative_spend_dollars", 0.0)
    cum_wasted = fin.get("cumulative_wasted_dollars", 0.0)
    cum_time_wasted = fin.get("cumulative_time_wasted_sec", 0.0)

    code = diag.get("code", "UNKNOWN")
    desc = diag.get("description", "No diagnosis available.")

    # Status indicator
    if "SATURATED" in code or "OPTIMAL" in code:
        status_badge = "🟢 HIGH EFFICIENCY"
    elif "BALANCED" in code:
        status_badge = "⚡ BALANCED LOAD"
    elif "BOTTLENECK" in code:
        status_badge = "⚠️ CPU BOTTLENECK"
    else:
        status_badge = "🟡 MONITORING"

    print(f" Status: {status_badge:<22} │ Iteration: {iteration:<5} │ Last Pod Ping: {updated_at[:19]}")
    print("────────────────────────────────────────────────────────────────────────────────────")

    # Hardware Section
    print(" 🖥️  HARDWARE UTILIZATION")
    print(f"  GPU [{gpu_name[:26]:<26}]  {render_bar(gpu_util)} (Peak: {gpu_peak:.1f}%)")
    print(f"  VRAM Memory ({vram_used:>5.0f} / {vram_total:>5.0f} MB)  {render_bar(vram_pct)} Power: {power:.0f}W")
    print(f"  CPU Processing Cores           {render_bar(cpu_util)} (Peak: {cpu_peak:.1f}%)")
    print(f"  System RAM Usage               {ram_used:.0f} MB active")

    print("────────────────────────────────────────────────────────────────────────────────────")
    # Throughput Section
    print(" ⚡ SIMULATION VELOCITY")
    print(f"  Throughput:  {moves_sec:>8.1f} moves/second")
    print(f"  Game Volume: {games_count:>8} games per iteration ({duration:.1f}s iteration duration)")

    print("────────────────────────────────────────────────────────────────────────────────────")
    # Financial & Waste Section
    print(f" 💸 FINANCIAL & TIME WASTE AUDIT (Billing Rate: ${rate:.2f}/hr)")
    print(f"  Iteration Time Wasted:  {time_wasted:>5.1f}s idle / {duration:>5.1f}s total ({waste_pct:.1f}% wasted)")
    print(f"  Iteration Money Wasted: ${iter_wasted_money:>6.4f} burned on idle cycles (Cost: ${iter_cost:.4f})")
    print(f"  Cumulative Run Spend:   ${cum_spend:>6.2f} total  │ Cumulative Wasted: ${cum_wasted:>6.2f}")
    if cum_time_wasted > 0:
        print(f"  Cumulative Time Lost:   {format_duration(cum_time_wasted)} total idle time on pod")
    if vram_waste > 75.0:
        print(f"  Memory Overprovision:   {vram_waste:.1f}% excess VRAM (paying for tier you don't need)")

    print("────────────────────────────────────────────────────────────────────────────────────")
    # Diagnostic Verdict
    print(f" 🎯 DIAGNOSTIC VERDICT: {code}")
    print(f"  {desc}")
    print("────────────────────────────────────────────────────────────────────────────────────")
    print("  Streaming live from Supabase... Press Ctrl+C to stop.")


def main():
    parser = argparse.ArgumentParser(description="Live Telemetry Dashboard for PolyZero RunPod")
    parser.add_argument("--interval", type=float, default=3.0, help="Refresh interval in seconds (default: 3.0)")
    parser.add_argument("--oneshot", action="store_true", help="Print current status once and exit")
    parser.add_argument("--bucket", default="models", help="Supabase storage bucket (default: models)")
    args = parser.parse_args()

    if args.oneshot:
        data = fetch_live_status(bucket=args.bucket)
        if not data:
            print("No live telemetry found in Supabase. Pod may be offline or initializing.")
            sys.exit(1)
        render_dashboard(data)
        return

    try:
        while True:
            err = None
            data = None
            try:
                data = fetch_live_status(bucket=args.bucket)
            except Exception as e:
                err = str(e)

            render_dashboard(data, error_msg=err)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n\nMonitor stopped. Pod is continuing training in the cloud.\n")


if __name__ == "__main__":
    main()
