#!/usr/bin/env python3
"""
coordinator.py — Distributed training coordinator using Supabase Postgres.

Coordinates multiple RunPod GPU pods sharing an NFS volume to run
self-play (Rust binary) and training (train.py) in a loop.

Usage:
    python3 coordinator.py <command> [args]

Commands:
    ensure-tables                                   Print SQL to create tables (run in Supabase SQL editor)
    register-pod                                    Register this pod
    heartbeat-loop                                  Run persistent heartbeat (launch as bg process)
    plan-iteration <iter> <run_id> <total_games>    Create iteration + distribute games
    wait-for-assignment <iter>                       Wait for selfplay task assignment
    complete-selfplay <iter> <task_id> <games_file> <metrics_json>
    acquire-training-lock <iter>                     Atomically acquire training lock
    wait-for-training <iter>                         Wait until training is complete
    release-iteration <iter> [metrics_json]          Mark iteration complete
    pod-status                                       Show all pods
    cleanup                                          Mark pod offline
"""

import json
import os
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone, timedelta

try:
    from supabase import create_client, Client
except ImportError:
    print("ERROR: supabase package not installed. Run: pip install supabase")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

_heartbeat_stop = threading.Event()
_heartbeat_thread = None


def get_pod_id() -> str:
    """Pod ID from POD_ID env var, falling back to hostname."""
    return os.environ.get("POD_ID", socket.gethostname())


def get_client() -> Client:
    """Load .env and return a Supabase client (same pattern as supabase_sync.py)."""
    from dotenv import load_dotenv
    load_dotenv()
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        print("ERROR: Missing SUPABASE_URL or SUPABASE_KEY / SUPABASE_SERVICE_ROLE_KEY in .env")
        sys.exit(1)
    return create_client(url, key)


def get_gpu_name() -> str:
    """Best-effort GPU name via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, timeout=5
        )
        return out.decode().strip().split("\n")[0]
    except Exception:
        return "unknown"


def utcnow_iso() -> str:
    """UTC now as ISO-8601 string (Supabase-friendly)."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# 1. ensure-tables
# ---------------------------------------------------------------------------

TABLE_SQL = r"""
-- ==========================================================================
-- Polyfish Coordinator Tables
-- Run this SQL in your Supabase SQL Editor (https://supabase.com/dashboard)
-- ==========================================================================

CREATE TABLE IF NOT EXISTS pods (
    pod_id TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'online',
    gpu_name TEXT,
    last_heartbeat TIMESTAMPTZ NOT NULL DEFAULT now(),
    joined_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    current_task TEXT,
    games_assigned INT DEFAULT 0,
    games_completed INT DEFAULT 0,
    meta JSONB DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS iterations (
    iteration INT PRIMARY KEY,
    run_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    total_games INT NOT NULL,
    games_completed INT DEFAULT 0,
    trainer_pod TEXT,
    created_at TIMESTAMPTZ DEFAULT now(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    train_metrics JSONB,
    selfplay_metrics JSONB,
    meta JSONB DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS selfplay_tasks (
    id SERIAL PRIMARY KEY,
    iteration INT NOT NULL,
    pod_id TEXT NOT NULL,
    num_games INT NOT NULL,
    status TEXT NOT NULL DEFAULT 'assigned',
    games_file TEXT,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    metrics JSONB DEFAULT '{}',
    UNIQUE(iteration, pod_id)
);

-- Index for fast lookups of online pods
CREATE INDEX IF NOT EXISTS idx_pods_status ON pods(status);
CREATE INDEX IF NOT EXISTS idx_pods_heartbeat ON pods(last_heartbeat);

-- Index for task queries
CREATE INDEX IF NOT EXISTS idx_selfplay_iteration ON selfplay_tasks(iteration);
CREATE INDEX IF NOT EXISTS idx_selfplay_pod ON selfplay_tasks(pod_id);
"""


def cmd_ensure_tables():
    """Print the SQL that must be run in the Supabase SQL editor."""
    print("=" * 72)
    print("IMPORTANT: Copy and run the following SQL in your Supabase SQL Editor")
    print("           https://supabase.com/dashboard → SQL Editor → New Query")
    print("=" * 72)
    print(TABLE_SQL)
    print("=" * 72)
    print("After running the SQL, the coordinator is ready to use.")


# ---------------------------------------------------------------------------
# 2. register-pod
# ---------------------------------------------------------------------------

def _heartbeat_loop(sb: Client, pod_id: str, interval: float = 15.0):
    """Background daemon thread that sends heartbeats every `interval` seconds."""
    while not _heartbeat_stop.is_set():
        try:
            sb.table("pods").update({
                "last_heartbeat": utcnow_iso(),
                "status": "online",
            }).eq("pod_id", pod_id).execute()
        except Exception as e:
            # Silently retry — transient network issues are expected
            print(f"⚠️  Heartbeat failed (will retry): {e}", file=sys.stderr)
        _heartbeat_stop.wait(interval)


def cmd_register_pod():
    """Upsert this pod into the pods table and start a heartbeat daemon."""
    global _heartbeat_thread
    sb = get_client()
    pod_id = get_pod_id()
    gpu = get_gpu_name()
    now = utcnow_iso()

    try:
        sb.table("pods").upsert({
            "pod_id": pod_id,
            "status": "online",
            "gpu_name": gpu,
            "last_heartbeat": now,
            "joined_at": now,
            "current_task": None,
            "games_assigned": 0,
            "games_completed": 0,
            "meta": {},
        }, on_conflict="pod_id").execute()
    except Exception as e:
        print(f"ERROR: Failed to register pod: {e}", file=sys.stderr)
        sys.exit(1)

    # Start heartbeat daemon
    _heartbeat_stop.clear()
    _heartbeat_thread = threading.Thread(
        target=_heartbeat_loop, args=(sb, pod_id), daemon=True
    )
    _heartbeat_thread.start()

    print(pod_id)


# ---------------------------------------------------------------------------
# 3. plan-iteration
# ---------------------------------------------------------------------------

def _get_online_pods(sb: Client, stale_seconds: int = 60) -> list[dict]:
    """Return pods whose heartbeat is within `stale_seconds` of now."""
    cutoff = (datetime.now(timezone.utc) - timedelta(seconds=stale_seconds)).isoformat()
    try:
        resp = sb.table("pods") \
            .select("pod_id") \
            .eq("status", "online") \
            .gte("last_heartbeat", cutoff) \
            .execute()
        return resp.data or []
    except Exception as e:
        print(f"ERROR: Failed to query online pods: {e}", file=sys.stderr)
        return []


def cmd_plan_iteration(iteration: int, run_id: str, total_games: int):
    """Create iteration row + distribute selfplay games across online pods."""
    sb = get_client()

    # 1. Create iteration row (idempotent via ON CONFLICT DO NOTHING / upsert-ignore)
    try:
        # Try insert; if it already exists, this is a no-op
        existing = sb.table("iterations") \
            .select("iteration") \
            .eq("iteration", iteration) \
            .execute()

        if not existing.data:
            sb.table("iterations").insert({
                "iteration": iteration,
                "run_id": run_id,
                "status": "selfplay",
                "total_games": total_games,
                "games_completed": 0,
                "started_at": utcnow_iso(),
                "meta": {},
            }).execute()
        else:
            # Already planned — still distribute tasks in case pods changed
            pass
    except Exception as e:
        # Duplicate key = already planned, which is fine
        if "duplicate" in str(e).lower() or "23505" in str(e):
            pass
        else:
            print(f"ERROR: Failed to create iteration row: {e}", file=sys.stderr)
            sys.exit(1)

    # 2. Find all online pods
    online = _get_online_pods(sb)
    if not online:
        print("ERROR: No online pods found. Register pods first.", file=sys.stderr)
        sys.exit(1)

    pod_ids = [p["pod_id"] for p in online]
    pod_count = len(pod_ids)

    # 3. Divide games evenly; remainder goes to the first pods
    base = total_games // pod_count
    remainder = total_games % pod_count
    games_per_pod = {}
    for i, pid in enumerate(sorted(pod_ids)):
        games_per_pod[pid] = base + (1 if i < remainder else 0)

    # 4. Create selfplay_tasks rows (idempotent: skip if task already exists)
    for pid, num in games_per_pod.items():
        if num == 0:
            continue
        try:
            # Check if task already exists for this (iteration, pod_id)
            existing_task = sb.table("selfplay_tasks") \
                .select("id") \
                .eq("iteration", iteration) \
                .eq("pod_id", pid) \
                .execute()

            if not existing_task.data:
                sb.table("selfplay_tasks").insert({
                    "iteration": iteration,
                    "pod_id": pid,
                    "num_games": num,
                    "status": "assigned",
                    "started_at": utcnow_iso(),
                    "metrics": {},
                }).execute()
        except Exception as e:
            if "duplicate" in str(e).lower() or "23505" in str(e):
                pass  # Already assigned — idempotent
            else:
                print(f"⚠️  Failed to create task for {pid}: {e}", file=sys.stderr)

    # 5. Update pods table with assignment info
    for pid, num in games_per_pod.items():
        try:
            sb.table("pods").update({
                "current_task": f"selfplay_iter{iteration}",
                "games_assigned": num,
                "games_completed": 0,
            }).eq("pod_id", pid).execute()
        except Exception:
            pass  # Best-effort update

    result = {"pod_count": pod_count, "games_per_pod": games_per_pod}
    print(json.dumps(result))


# ---------------------------------------------------------------------------
# 4. wait-for-assignment
# ---------------------------------------------------------------------------

def cmd_wait_for_assignment(iteration: int):
    """Poll for a selfplay task assigned to this pod. Timeout 120s."""
    sb = get_client()
    pod_id = get_pod_id()
    timeout = 120
    poll_interval = 2
    start = time.time()

    while time.time() - start < timeout:
        try:
            # Check for a task directly assigned to this pod
            resp = sb.table("selfplay_tasks") \
                .select("id,num_games,status") \
                .eq("iteration", iteration) \
                .eq("pod_id", pod_id) \
                .in_("status", ["assigned", "reassigned"]) \
                .execute()

            if resp.data:
                task = resp.data[0]
                result = {"num_games": task["num_games"], "task_id": task["id"]}
                print(json.dumps(result))
                return

            # Also check for tasks from dead pods that could be reassigned
            # Find tasks assigned to pods that have gone stale
            cutoff = (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat()
            stale_tasks = sb.table("selfplay_tasks") \
                .select("id,pod_id,num_games") \
                .eq("iteration", iteration) \
                .eq("status", "assigned") \
                .execute()

            if stale_tasks.data:
                for task in stale_tasks.data:
                    if task["pod_id"] == pod_id:
                        continue  # Already checked above
                    # Check if the assigned pod is dead
                    pod_resp = sb.table("pods") \
                        .select("last_heartbeat") \
                        .eq("pod_id", task["pod_id"]) \
                        .execute()

                    if pod_resp.data:
                        hb = pod_resp.data[0]["last_heartbeat"]
                        if hb and hb < cutoff:
                            # Pod is dead — try to reassign this task to us
                            try:
                                sb.table("selfplay_tasks").update({
                                    "pod_id": pod_id,
                                    "status": "reassigned",
                                    "started_at": utcnow_iso(),
                                }).eq("id", task["id"]).eq("status", "assigned").execute()

                                result = {"num_games": task["num_games"], "task_id": task["id"]}
                                print(json.dumps(result))
                                return
                            except Exception:
                                pass  # Another pod may have grabbed it first

        except Exception as e:
            print(f"⚠️  Poll error (will retry): {e}", file=sys.stderr)

        time.sleep(poll_interval)

    print("ERROR: Timed out waiting for selfplay assignment", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# 5. complete-selfplay
# ---------------------------------------------------------------------------

def cmd_complete_selfplay(iteration: int, task_id: int, games_file: str, metrics_json: str):
    """Mark a selfplay task as complete with results."""
    sb = get_client()
    pod_id = get_pod_id()

    try:
        metrics = json.loads(metrics_json)
    except json.JSONDecodeError:
        metrics = {"raw": metrics_json}

    try:
        sb.table("selfplay_tasks").update({
            "status": "done",
            "games_file": games_file,
            "completed_at": utcnow_iso(),
            "metrics": metrics,
        }).eq("id", task_id).execute()
    except Exception as e:
        print(f"ERROR: Failed to mark task complete: {e}", file=sys.stderr)
        sys.exit(1)

    # Update pod status
    try:
        sb.table("pods").update({
            "current_task": None,
            "games_completed": (
                sb.table("selfplay_tasks")
                .select("num_games")
                .eq("id", task_id)
                .execute()
                .data[0]["num_games"]
            ),
        }).eq("pod_id", pod_id).execute()
    except Exception:
        pass  # Best-effort

    # Update iteration games_completed count
    try:
        done_tasks = sb.table("selfplay_tasks") \
            .select("num_games") \
            .eq("iteration", iteration) \
            .eq("status", "done") \
            .execute()
        total_done = sum(t["num_games"] for t in (done_tasks.data or []))

        sb.table("iterations").update({
            "games_completed": total_done,
        }).eq("iteration", iteration).execute()
    except Exception:
        pass  # Best-effort

    print(json.dumps({"status": "ok", "task_id": task_id}))


# ---------------------------------------------------------------------------
# 6. acquire-training-lock
# ---------------------------------------------------------------------------

def cmd_acquire_training_lock(iteration: int):
    """
    Atomically try to become the trainer for this iteration.

    Exit code 0 = got the lock, exit code 1 = didn't get it.

    Strategy:
      1. Check all selfplay_tasks for this iteration are 'done'
      2. UPDATE iterations SET trainer_pod=me, status='training'
         WHERE iteration=X AND status='selfplay' AND trainer_pod IS NULL
      3. Only one pod can match all conditions
    """
    sb = get_client()
    pod_id = get_pod_id()

    # Step 1: Verify all selfplay tasks are done
    try:
        tasks = sb.table("selfplay_tasks") \
            .select("id,status") \
            .eq("iteration", iteration) \
            .execute()

        if not tasks.data:
            print("No selfplay tasks found for this iteration", file=sys.stderr)
            sys.exit(1)

        not_done = [t for t in tasks.data if t["status"] != "done"]
        if not_done:
            print(f"Selfplay not complete: {len(not_done)} tasks remaining", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to check task status: {e}", file=sys.stderr)
        sys.exit(1)

    # Step 2: Atomic lock — update only if status='selfplay' and trainer_pod is null
    try:
        resp = sb.table("iterations").update({
            "trainer_pod": pod_id,
            "status": "training",
            "started_at": utcnow_iso(),
        }).eq("iteration", iteration) \
          .eq("status", "selfplay") \
          .is_("trainer_pod", "null") \
          .execute()

        if resp.data and len(resp.data) > 0:
            # We got the lock
            sb.table("pods").update({
                "current_task": f"training_iter{iteration}",
            }).eq("pod_id", pod_id).execute()
            print(json.dumps({"status": "acquired", "trainer_pod": pod_id}))
            sys.exit(0)
        else:
            # Someone else got it, or state already changed
            print("Lock not acquired (another pod is training)", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to acquire training lock: {e}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# 7. wait-for-training
# ---------------------------------------------------------------------------

def cmd_wait_for_training(iteration: int):
    """Poll iterations table until status='complete'. Timeout 3600s."""
    sb = get_client()
    timeout = 3600
    poll_interval = 5
    start = time.time()

    while time.time() - start < timeout:
        try:
            resp = sb.table("iterations") \
                .select("status,trainer_pod,train_metrics") \
                .eq("iteration", iteration) \
                .execute()

            if not resp.data:
                print(f"⚠️  Iteration {iteration} not found, waiting...", file=sys.stderr)
                time.sleep(poll_interval)
                continue

            row = resp.data[0]

            if row["status"] == "complete":
                print(json.dumps({
                    "status": "complete",
                    "trainer_pod": row["trainer_pod"],
                    "train_metrics": row.get("train_metrics"),
                }))
                return

            if row["status"] == "failed":
                print("ERROR: Training failed", file=sys.stderr)
                sys.exit(1)

            # Check if trainer pod is dead
            trainer = row.get("trainer_pod")
            if trainer and row["status"] == "training":
                cutoff = (datetime.now(timezone.utc) - timedelta(seconds=90)).isoformat()
                pod_resp = sb.table("pods") \
                    .select("last_heartbeat") \
                    .eq("pod_id", trainer) \
                    .execute()

                if pod_resp.data:
                    hb = pod_resp.data[0]["last_heartbeat"]
                    if hb and hb < cutoff:
                        print(f"⚠️  Trainer pod '{trainer}' appears dead (stale heartbeat). "
                              f"Resetting iteration to selfplay state.", file=sys.stderr)
                        # Reset the iteration so another pod can grab the lock
                        try:
                            sb.table("iterations").update({
                                "trainer_pod": None,
                                "status": "selfplay",
                            }).eq("iteration", iteration) \
                              .eq("status", "training") \
                              .eq("trainer_pod", trainer) \
                              .execute()
                        except Exception:
                            pass
                        # Return with error so the calling script can retry
                        print("ERROR: Trainer pod died, iteration reset", file=sys.stderr)
                        sys.exit(2)

        except Exception as e:
            print(f"⚠️  Poll error (will retry): {e}", file=sys.stderr)

        time.sleep(poll_interval)

    print("ERROR: Timed out waiting for training to complete", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# 8. release-iteration
# ---------------------------------------------------------------------------

def cmd_release_iteration(iteration: int, metrics_json: str | None = None):
    """Mark iteration as complete. Called by the trainer pod after train.py."""
    sb = get_client()
    pod_id = get_pod_id()

    metrics = None
    if metrics_json:
        try:
            metrics = json.loads(metrics_json)
        except json.JSONDecodeError:
            metrics = {"raw": metrics_json}

    # Gather selfplay metrics from all tasks
    selfplay_metrics = None
    try:
        tasks = sb.table("selfplay_tasks") \
            .select("pod_id,num_games,metrics,completed_at") \
            .eq("iteration", iteration) \
            .eq("status", "done") \
            .execute()
        if tasks.data:
            selfplay_metrics = {
                "tasks": tasks.data,
                "total_games": sum(t["num_games"] for t in tasks.data),
            }
    except Exception:
        pass

    update_data = {
        "status": "complete",
        "completed_at": utcnow_iso(),
    }
    if metrics is not None:
        update_data["train_metrics"] = metrics
    if selfplay_metrics is not None:
        update_data["selfplay_metrics"] = selfplay_metrics

    try:
        sb.table("iterations").update(update_data) \
            .eq("iteration", iteration) \
            .execute()
    except Exception as e:
        print(f"ERROR: Failed to release iteration: {e}", file=sys.stderr)
        sys.exit(1)

    # Clear pod task
    try:
        sb.table("pods").update({
            "current_task": None,
        }).eq("pod_id", pod_id).execute()
    except Exception:
        pass

    print(json.dumps({"status": "complete", "iteration": iteration}))


# ---------------------------------------------------------------------------
# 9. pod-status
# ---------------------------------------------------------------------------

def cmd_pod_status():
    """Print JSON status of all pods."""
    sb = get_client()

    try:
        resp = sb.table("pods") \
            .select("*") \
            .order("pod_id") \
            .execute()

        pods = resp.data or []
        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat()

        for p in pods:
            hb = p.get("last_heartbeat", "")
            p["is_alive"] = bool(hb and hb >= cutoff)

        print(json.dumps({"pods": pods, "total": len(pods)}, indent=2, default=str))
    except Exception as e:
        print(f"ERROR: Failed to fetch pod status: {e}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# 10a. heartbeat-loop (persistent background process)
# ---------------------------------------------------------------------------

def cmd_heartbeat_loop():
    """Run a persistent heartbeat loop. Launch as a background process from bash.

    Usage from shell:
        .venv/bin/python3 coordinator.py heartbeat-loop &
        HEARTBEAT_PID=$!
    """
    sb = get_client()
    pod_id = get_pod_id()
    interval = 15.0

    import signal
    running = True

    def _stop(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    while running:
        try:
            sb.table("pods").update({
                "last_heartbeat": utcnow_iso(),
                "status": "online",
            }).eq("pod_id", pod_id).execute()
        except Exception as e:
            print(f"⚠️  Heartbeat failed (will retry): {e}", file=sys.stderr, flush=True)
        time.sleep(interval)

    # On exit, mark pod offline
    try:
        sb.table("pods").update({
            "status": "offline",
            "current_task": None,
        }).eq("pod_id", pod_id).execute()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 10b. cleanup
# ---------------------------------------------------------------------------

def cmd_cleanup():
    """Mark this pod as offline and stop heartbeat."""
    global _heartbeat_thread
    sb = get_client()
    pod_id = get_pod_id()

    # Stop heartbeat
    _heartbeat_stop.set()
    if _heartbeat_thread and _heartbeat_thread.is_alive():
        _heartbeat_thread.join(timeout=5)

    try:
        sb.table("pods").update({
            "status": "offline",
            "current_task": None,
            "last_heartbeat": utcnow_iso(),
        }).eq("pod_id", pod_id).execute()
        print(json.dumps({"status": "offline", "pod_id": pod_id}))
    except Exception as e:
        print(f"ERROR: Failed to mark pod offline: {e}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI Dispatch
# ---------------------------------------------------------------------------

COMMANDS = {
    "ensure-tables":        (cmd_ensure_tables, 0, 0),
    "register-pod":         (cmd_register_pod, 0, 0),
    "plan-iteration":       (lambda a: cmd_plan_iteration(int(a[0]), a[1], int(a[2])), 3, 3),
    "wait-for-assignment":  (lambda a: cmd_wait_for_assignment(int(a[0])), 1, 1),
    "complete-selfplay":    (lambda a: cmd_complete_selfplay(int(a[0]), int(a[1]), a[2], a[3]), 4, 4),
    "acquire-training-lock":(lambda a: cmd_acquire_training_lock(int(a[0])), 1, 1),
    "wait-for-training":    (lambda a: cmd_wait_for_training(int(a[0])), 1, 1),
    "release-iteration":    (lambda a: cmd_release_iteration(int(a[0]), a[1] if len(a) > 1 else None), 1, 2),
    "pod-status":           (cmd_pod_status, 0, 0),
    "heartbeat-loop":       (cmd_heartbeat_loop, 0, 0),
    "cleanup":              (cmd_cleanup, 0, 0),
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__.strip())
        sys.exit(0)

    cmd_name = sys.argv[1]
    args = sys.argv[2:]

    if cmd_name not in COMMANDS:
        print(f"ERROR: Unknown command '{cmd_name}'")
        print(f"Available commands: {', '.join(COMMANDS.keys())}")
        sys.exit(1)

    func, min_args, max_args = COMMANDS[cmd_name]

    if len(args) < min_args or len(args) > max_args:
        if min_args == max_args:
            print(f"ERROR: '{cmd_name}' requires exactly {min_args} argument(s), got {len(args)}")
        else:
            print(f"ERROR: '{cmd_name}' requires {min_args}-{max_args} argument(s), got {len(args)}")
        sys.exit(1)

    # Dispatch: commands with 0 args are called directly, others get the args list
    if max_args == 0:
        func()
    else:
        func(args)


if __name__ == "__main__":
    main()
