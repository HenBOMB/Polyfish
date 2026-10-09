#!/usr/bin/env python3
"""Hardware Telemetry, Time-Wasted Tracking, and Financial Analytics for PolyZero.

Tracks GPU utilization, VRAM, power, CPU intensity, simulation throughput,
and explicitly calculates:
    1. Time Wasted (seconds the GPU/system sat idling waiting on CPU bottlenecks).
    2. Money Wasted (exact dollar amount paid for idle/unproductive compute,
       both per iteration and cumulatively across the run).

Usage:
    # Probe hardware:
    python telemetry.py probe

    # Record during training loop:
    python telemetry.py start
    # ... run self_play ...
    python telemetry.py record --iteration 76 --moves-per-sec 1420.5 --games 50
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

TELEMETRY_CSV = "hardware_telemetry.csv"
TELEMETRY_JSON = "hardware_status.json"
PID_FILE = ".telemetry_sampler.pid"
SAMPLES_FILE = ".telemetry_samples.json"
CUMULATIVE_FILE = ".telemetry_cumulative.json"

# Typical RunPod on-demand market pricing ($/hr) for automatic cost estimation
GPU_MARKET_RATES = {
    "H100": 3.49,
    "A100": 1.89,
    "L40S": 0.84,
    "L40": 0.84,
    "A40": 0.74,
    "4090": 0.69,
    "3090": 0.34,
    "4080": 0.54,
    "A5000": 0.44,
    "A6000": 0.79,
    "3080": 0.29,
    "V100": 0.49,
}


def estimate_hourly_rate(gpu_name: str) -> float:
    """Estimate RunPod hourly cost ($/hr) based on detected GPU name if not set."""
    name_upper = gpu_name.upper()
    for model_key, rate in GPU_MARKET_RATES.items():
        if model_key in name_upper:
            return rate
    return 0.50


def get_cpu_info() -> Dict[str, Any]:
    """Inspect CPU model and core counts."""
    count = os.cpu_count() or 1
    model = "Unknown CPU"
    try:
        if os.path.exists("/proc/cpuinfo"):
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        model = line.split(":", 1)[1].strip()
                        break
    except Exception:
        pass
    return {"cores": count, "model": model}


def get_ram_info() -> Dict[str, float]:
    """Inspect system RAM in MB."""
    total_mb = 0.0
    free_mb = 0.0
    try:
        if os.path.exists("/proc/meminfo"):
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        total_mb = float(line.split()[1]) / 1024.0
                    elif line.startswith("MemAvailable:"):
                        free_mb = float(line.split()[1]) / 1024.0
    except Exception:
        pass
    return {"total_mb": total_mb, "used_mb": max(0.0, total_mb - free_mb)}


def query_nvidia_smi() -> List[Dict[str, Any]]:
    """Query GPU utilization, VRAM, power, and temperature via nvidia-smi."""
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu,utilization.memory,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=2.0)
        gpus = []
        for line in res.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 9:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "driver": parts[2],
                    "mem_total_mb": float(parts[3]),
                    "mem_used_mb": float(parts[4]),
                    "gpu_util_pct": float(parts[5]),
                    "mem_util_pct": float(parts[6]),
                    "power_watts": float(parts[7]) if parts[7] != "[N/A]" else 0.0,
                    "temp_c": float(parts[8]) if parts[8] != "[N/A]" else 0.0,
                })
        return gpus
    except Exception:
        return []


def read_proc_stat_cpu() -> tuple[int, int]:
    """Read aggregate CPU ticks (work, total) from /proc/stat."""
    try:
        with open("/proc/stat", "r") as f:
            for line in f:
                if line.startswith("cpu "):
                    parts = [int(p) for p in line.split()[1:8]]
                    idle = parts[3] + (parts[4] if len(parts) > 4 else 0)
                    total = sum(parts)
                    work = total - idle
                    return work, total
    except Exception:
        pass
    return 0, 0


class SamplerProcess:
    """Background sampling loop writing time-series metrics to SAMPLES_FILE."""

    def __init__(self, interval_sec: float = 1.5):
        self.interval_sec = interval_sec
        self.stop_event = threading.Event()
        self.samples: List[Dict[str, Any]] = []

    def run(self):
        last_work, last_total = read_proc_stat_cpu()
        start_time = time.time()

        while not self.stop_event.is_set():
            time.sleep(self.interval_sec)
            now = time.time()

            # CPU
            cur_work, cur_total = read_proc_stat_cpu()
            d_work = cur_work - last_work
            d_total = cur_total - last_total
            cpu_pct = (100.0 * d_work / max(1, d_total)) if d_total > 0 else 0.0
            last_work, last_total = cur_work, cur_total

            # RAM
            ram = get_ram_info()

            # GPU
            gpus = query_nvidia_smi()
            primary_gpu = gpus[0] if gpus else {}

            sample = {
                "t": now - start_time,
                "cpu_util_pct": round(cpu_pct, 1),
                "ram_used_mb": round(ram["used_mb"], 1),
                "gpu_util_pct": primary_gpu.get("gpu_util_pct", 0.0),
                "gpu_mem_used_mb": primary_gpu.get("mem_used_mb", 0.0),
                "gpu_mem_total_mb": primary_gpu.get("mem_total_mb", 0.0),
                "gpu_power_watts": primary_gpu.get("power_watts", 0.0),
                "gpu_temp_c": primary_gpu.get("temp_c", 0.0),
                "gpu_name": primary_gpu.get("name", "None / CPU"),
            }
            self.samples.append(sample)
            try:
                with open(SAMPLES_FILE, "w") as f:
                    json.dump(self.samples, f)
            except Exception:
                pass


def diagnose_bottleneck(gpu_avg: float, cpu_avg: float, vram_used: float, vram_total: float) -> tuple[str, str]:
    """Determine whether the training iteration was CPU-bound, GPU-bound, or wasted money."""
    if gpu_avg == 0.0 and vram_total == 0.0:
        return "CPU_ONLY", "Running purely on CPU. No CUDA GPU active."

    if gpu_avg < 35.0 and cpu_avg > 85.0:
        return (
            "CPU_BOTTLENECK",
            "⚠️ Severe CPU throttle. GPU is idling (<35%) while CPU is pegged (>85%). "
            "You are overpaying for GPU compute! Upgrade to faster/more CPU cores, or switch to RTX 4090/3090.",
        )
    elif gpu_avg >= 80.0:
        return (
            "GPU_SATURATED",
            "🏆 Optimal. GPU compute is highly saturated (>=80%). Highest throughput per dollar.",
        )
    elif gpu_avg >= 50.0:
        return (
            "BALANCED",
            "⚡ Balanced execution. GPU is doing real work (50-80%). Increase --actors (-a) to push it higher.",
        )
    elif gpu_avg < 40.0 and cpu_avg < 50.0:
        return (
            "UNDER_SUBSCRIBED",
            "⚠️ Under-utilized. Both CPU and GPU are low (<50%). Increase parallel games (-g) or actors (-a).",
        )
    else:
        return "MODERATE", "Moderate load. GPU and CPU shared evenly."


def load_cumulative_stats() -> Dict[str, float]:
    """Load running totals across the whole training run."""
    if os.path.exists(CUMULATIVE_FILE):
        try:
            with open(CUMULATIVE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "total_wall_sec": 0.0,
        "total_time_wasted_sec": 0.0,
        "total_spend_dollars": 0.0,
        "total_wasted_dollars": 0.0,
        "total_games": 0.0,
        "iterations_counted": 0.0,
    }


def save_cumulative_stats(stats: Dict[str, float]):
    """Save running totals to CUMULATIVE_FILE."""
    try:
        with open(CUMULATIVE_FILE, "w") as f:
            json.dump(stats, f, indent=2)
    except Exception:
        pass


def cmd_probe():
    """Print hardware specs, estimated hourly rate, and utilization immediately."""
    cpu = get_cpu_info()
    ram = get_ram_info()
    gpus = query_nvidia_smi()

    rate = estimate_hourly_rate(gpus[0]["name"]) if gpus else 0.0
    user_rate = os.environ.get("POD_HOURLY_COST")
    if user_rate:
        try:
            rate = float(user_rate)
        except ValueError:
            pass

    print("==============================================================================")
    print(" 🖥️  HARDWARE PROBE & FINANCIAL AUDIT")
    print("==============================================================================")
    print(f" CPU:   {cpu['model']} ({cpu['cores']} logical cores)")
    print(f" RAM:   {ram['used_mb']:.0f} MB used / {ram['total_mb']:.0f} MB total ({ram['used_mb']/max(1, ram['total_mb'])*100:.1f}%)")
    if gpus:
        for g in gpus:
            print(f" GPU {g['index']}: {g['name']} (Driver: {g['driver']})")
            print(f"        VRAM:  {g['mem_used_mb']:.0f} MB / {g['mem_total_mb']:.0f} MB ({g['mem_util_pct']}%)")
            print(f"        Load:  Compute {g['gpu_util_pct']}% | Power {g['power_watts']}W | Temp {g['temp_c']}°C")
        print(f" Pod Pricing: ~${rate:.2f}/hr (Set via POD_HOURLY_COST to customize)")
    else:
        print(" GPU:   None detected via nvidia-smi (CPU-only execution).")
    print("==============================================================================")


def cmd_start():
    """Start background sampling process."""
    if os.path.exists(PID_FILE):
        try:
            pid = int(open(PID_FILE).read().strip())
            os.kill(pid, 0)
            return  # Already running
        except Exception:
            pass

    cmd = [sys.executable, os.path.abspath(__file__), "--daemon-worker"]
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setpgrp)
    with open(PID_FILE, "w") as f:
        f.write(str(p.pid))
    with open(SAMPLES_FILE, "w") as f:
        f.write("[]")


def cmd_daemon_worker():
    """Worker process that records samples until killed."""
    sampler = SamplerProcess(interval_sec=1.5)
    sampler.run()


def cmd_stop_and_record(
    iteration: int,
    moves_per_sec: float = 0.0,
    games_count: int = 0,
    hourly_cost: Optional[float] = None,
):
    """Stop sampler, aggregate metrics, calculate time & money wasted, and print audit card."""
    # Stop daemon
    if os.path.exists(PID_FILE):
        try:
            pid = int(open(PID_FILE).read().strip())
            os.kill(pid, 9)
        except Exception:
            pass
        try:
            os.remove(PID_FILE)
        except Exception:
            pass

    # Read samples
    samples = []
    if os.path.exists(SAMPLES_FILE):
        try:
            with open(SAMPLES_FILE, "r") as f:
                samples = json.load(f)
        except Exception:
            pass
        try:
            os.remove(SAMPLES_FILE)
        except Exception:
            pass

    if not samples:
        gpus = query_nvidia_smi()
        g = gpus[0] if gpus else {}
        ram = get_ram_info()
        samples = [{
            "cpu_util_pct": 50.0,
            "ram_used_mb": ram["used_mb"],
            "gpu_util_pct": g.get("gpu_util_pct", 0.0),
            "gpu_mem_used_mb": g.get("mem_used_mb", 0.0),
            "gpu_mem_total_mb": g.get("mem_total_mb", 0.0),
            "gpu_power_watts": g.get("power_watts", 0.0),
            "gpu_temp_c": g.get("temp_c", 0.0),
            "gpu_name": g.get("name", "N/A"),
        }]

    # Aggregations
    gpu_utils = [s["gpu_util_pct"] for s in samples]
    cpu_utils = [s["cpu_util_pct"] for s in samples]
    vram_used = [s["gpu_mem_used_mb"] for s in samples]
    powers = [s["gpu_power_watts"] for s in samples]

    gpu_avg = sum(gpu_utils) / len(gpu_utils)
    gpu_max = max(gpu_utils)
    cpu_avg = sum(cpu_utils) / len(cpu_utils)
    cpu_max = max(cpu_utils)
    vram_avg = sum(vram_used) / len(vram_used)
    vram_max = max(vram_used)
    vram_total = samples[-1].get("gpu_mem_total_mb", 0.0)
    power_avg = sum(powers) / len(powers)
    gpu_name = samples[-1].get("gpu_name", "Unknown GPU")
    ram_used = samples[-1].get("ram_used_mb", 0.0)

    duration_sec = samples[-1]["t"] if "t" in samples[-1] and samples[-1]["t"] > 0 else 1.0

    # Pricing resolution
    if hourly_cost is None or hourly_cost <= 0.0:
        env_cost = os.environ.get("POD_HOURLY_COST")
        if env_cost:
            try:
                hourly_cost = float(env_cost)
            except ValueError:
                hourly_cost = estimate_hourly_rate(gpu_name)
        else:
            hourly_cost = estimate_hourly_rate(gpu_name)

    # =========================================================================
    # Financial & Waste Math
    # =========================================================================
    # GPU Idle percentage during this iteration
    gpu_idle_pct = max(0.0, 100.0 - gpu_avg)
    time_wasted_sec = duration_sec * (gpu_idle_pct / 100.0)

    iteration_cost = hourly_cost * (duration_sec / 3600.0)
    money_wasted_iter = iteration_cost * (gpu_idle_pct / 100.0)

    cost_per_game = (iteration_cost / games_count) if games_count > 0 else 0.0
    wasted_per_game = (money_wasted_iter / games_count) if games_count > 0 else 0.0

    # Memory overprovision penalty: PolyZero requires ~3000 MB VRAM
    vram_excess_mb = max(0.0, vram_total - 3000.0)
    vram_waste_pct = (vram_excess_mb / max(1.0, vram_total)) * 100.0

    # Cumulative running totals
    cum = load_cumulative_stats()
    cum["total_wall_sec"] += duration_sec
    cum["total_time_wasted_sec"] += time_wasted_sec
    cum["total_spend_dollars"] += iteration_cost
    cum["total_wasted_dollars"] += money_wasted_iter
    cum["total_games"] += games_count
    cum["iterations_counted"] += 1
    save_cumulative_stats(cum)

    cum_waste_pct = (
        (cum["total_wasted_dollars"] / max(1e-5, cum["total_spend_dollars"])) * 100.0
        if cum["total_spend_dollars"] > 0
        else 0.0
    )

    verdict_code, verdict_desc = diagnose_bottleneck(gpu_avg, cpu_avg, vram_max, vram_total)

    # Print Terminal Diagnostic Card with explicit Time & Money Waste
    vram_pct = (vram_max / max(1.0, vram_total)) * 100.0
    print("")
    print("┌────────────────────────── 📊 HARDWARE & FINANCIAL AUDIT ──────────────────────────┐")
    print(f"│ Iteration: {iteration:<6} │ Duration: {duration_sec:>5.1f}s │ Speed: {moves_per_sec:>7.1f} moves/s ({games_count} games)     │")
    print("├──────────────────────────────────────────────────────────────────────────────────┤")
    print(f"│ GPU:  {gpu_name:<28} │ Load: Avg {gpu_avg:>5.1f}% (Peak {gpu_max:>5.1f}%)          │")
    print(f"│ VRAM: {vram_max:>6.0f} MB / {vram_total:>6.0f} MB ({vram_pct:>4.1f}%)   │ Power: {power_avg:>5.1f}W                       │")
    print(f"│ CPU:  Avg {cpu_avg:>5.1f}% (Peak {cpu_max:>5.1f}%)          │ System RAM: {ram_used:>6.0f} MB                       │")
    print("├──────────────────────────────────────────────────────────────────────────────────┤")
    print(f"│ ⏱️  TIME WASTED:   {time_wasted_sec:>5.1f}s idle / {duration_sec:>5.1f}s total ({gpu_idle_pct:>4.1f}% wasted idle time)      │")
    print(f"│ 💸 MONEY WASTED:  ${money_wasted_iter:>6.4f} this iter | ${cum['total_wasted_dollars']:>6.2f} cum. wasted (at ${hourly_cost:>4.2f}/hr)   │")
    print(f"│ 📉 CAPITAL WASTE: {gpu_idle_pct:>5.1f}% of your pod spend was burned on idle cycles            │")
    if vram_waste_pct > 75.0:
        print(f"│ 💾 VRAM OVERPAY:  {vram_waste_pct:>4.1f}% excess VRAM ({vram_excess_mb/1024:.0f}GB unneeded) - overpaying for tier!    │")
    print("├──────────────────────────────────────────────────────────────────────────────────┤")
    print(f"│ Diagnosis: {verdict_code:<15}                                                      │")
    print(f"│ {verdict_desc:<80} │")
    print("└──────────────────────────────────────────────────────────────────────────────────┘")
    print("")

    # CSV Logging
    csv_header = [
        "timestamp",
        "iteration",
        "duration_sec",
        "time_wasted_sec",
        "gpu_idle_pct",
        "moves_per_sec",
        "games_count",
        "gpu_name",
        "gpu_util_avg",
        "gpu_util_max",
        "vram_used_max_mb",
        "vram_total_mb",
        "vram_waste_pct",
        "gpu_power_avg_w",
        "cpu_util_avg",
        "cpu_util_max",
        "ram_used_mb",
        "bottleneck_code",
        "hourly_cost",
        "iter_cost_dollars",
        "iter_wasted_dollars",
        "cum_spend_dollars",
        "cum_wasted_dollars",
    ]
    file_exists = os.path.exists(TELEMETRY_CSV)
    with open(TELEMETRY_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(csv_header)
        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            iteration,
            round(duration_sec, 2),
            round(time_wasted_sec, 2),
            round(gpu_idle_pct, 2),
            round(moves_per_sec, 2),
            games_count,
            gpu_name,
            round(gpu_avg, 2),
            round(gpu_max, 2),
            round(vram_max, 1),
            round(vram_total, 1),
            round(vram_waste_pct, 1),
            round(power_avg, 1),
            round(cpu_avg, 2),
            round(cpu_max, 2),
            round(ram_used, 1),
            verdict_code,
            round(hourly_cost, 4),
            round(iteration_cost, 4),
            round(money_wasted_iter, 4),
            round(cum["total_spend_dollars"], 4),
            round(cum["total_wasted_dollars"], 4),
        ])

    # Status JSON update (live snapshot for dashboards and APIs)
    status_data = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "latest_iteration": iteration,
        "duration_sec": round(duration_sec, 2),
        "moves_per_sec": round(moves_per_sec, 2),
        "games_count": games_count,
        "financials": {
            "hourly_rate": round(hourly_cost, 2),
            "iteration_cost": round(iteration_cost, 4),
            "iteration_money_wasted": round(money_wasted_iter, 4),
            "time_wasted_sec": round(time_wasted_sec, 2),
            "waste_pct": round(gpu_idle_pct, 1),
            "cumulative_spend_dollars": round(cum["total_spend_dollars"], 2),
            "cumulative_wasted_dollars": round(cum["total_wasted_dollars"], 2),
            "cumulative_time_wasted_sec": round(cum["total_time_wasted_sec"], 1),
        },
        "gpu": {
            "name": gpu_name,
            "util_avg_pct": round(gpu_avg, 2),
            "util_max_pct": round(gpu_max, 2),
            "vram_used_mb": round(vram_max, 1),
            "vram_total_mb": round(vram_total, 1),
            "vram_util_pct": round(vram_pct, 1),
            "vram_waste_pct": round(vram_waste_pct, 1),
            "power_avg_w": round(power_avg, 1),
        },
        "cpu": {
            "util_avg_pct": round(cpu_avg, 2),
            "util_max_pct": round(cpu_max, 2),
            "ram_used_mb": round(ram_used, 1),
        },
        "diagnosis": {
            "code": verdict_code,
            "description": verdict_desc,
        },
    }
    with open(TELEMETRY_JSON, "w") as f:
        json.dump(status_data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="PolyZero Hardware Telemetry & Cost Analytics")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("probe", help="Probe current hardware specs and load")
    sub.add_parser("start", help="Start background sampler")
    sub.add_parser("--daemon-worker", help="Internal worker")

    rec = sub.add_parser("record", help="Stop sampler and record metrics")
    rec.add_argument("--iteration", type=int, required=True, help="Current iteration number")
    rec.add_argument("--moves-per-sec", type=float, default=0.0, help="Measured moves/sec")
    rec.add_argument("--games", type=int, default=0, help="Number of games in iteration")
    rec.add_argument("--cost-per-hour", type=float, default=None, help="Pod price $/hr")

    args = parser.parse_args()

    if args.cmd == "probe":
        cmd_probe()
    elif args.cmd == "start":
        cmd_start()
    elif args.cmd == "--daemon-worker":
        cmd_daemon_worker()
    elif args.cmd == "record":
        cmd_stop_and_record(
            iteration=args.iteration,
            moves_per_sec=args.moves_per_sec,
            games_count=args.games,
            hourly_cost=args.cost_per_hour,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
