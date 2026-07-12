import os
import sys
import json
import time
import glob
import shutil
import subprocess

# Ensure .venv/bin is in PATH so the kaggle CLI is found
os.environ["PATH"] = f"{os.path.abspath('.venv/bin')}:{os.environ['PATH']}"

# ==========================================
# Kaggle Integration Manager for Polyfish
# ==========================================
# Chained-kernel design: the trainer kernel keeps the model AND the replay
# window in its own output and reads them back on the next run via a
# kernel_sources self-reference. Each sync therefore uploads ONLY the fresh
# games staged in kaggle_pending/ (plus train.py) — not the model, not the
# archive. The first round ("bootstrap", before .kaggle_chain_ok exists)
# seeds the chain from the local model.safetensors and teacher games.
#
#   sync  -> version the dataset with pending games
#   train -> push the kernel and poll until complete
#   pull  -> download model.safetensors + last_train_metrics.json
#   all   -> sync, train, pull; on success clears kaggle_pending/ and
#            writes .kaggle_chain_ok

DATASET_NAME = "polyfish-training-data"
KERNEL_NAME = "polyfish-trainer"
DATA_DIR = "kaggle_data_sync"
KERNEL_DIR = "kaggle_kernel_sync"
PENDING_DIR = "kaggle_pending"
CHAIN_MARKER = ".kaggle_chain_ok"

REPLAY_WINDOW = int(os.environ.get("REPLAY_BUFFER_FILES", "10"))
DATASET_READY_TIMEOUT = 600
KERNEL_TIMEOUT = 3 * 3600
RETRIES = 3
RETRY_WAIT = 30


def run_cmd(cmd, retries=RETRIES, fatal=True):
    for attempt in range(1, retries + 1):
        suffix = f" (attempt {attempt}/{retries})" if attempt > 1 else ""
        #print(f"Running: {cmd}{suffix}")
        result = subprocess.run(cmd, shell=True, text=True, capture_output=True)
        if result.returncode == 0:
            return result.stdout.strip()
        print(f"[Kaggle] Error executing {cmd}:")
        print(result.stderr.strip() or result.stdout.strip())
        if attempt < retries:
            time.sleep(RETRY_WAIT)
    if fatal:
        sys.exit(1)
    return None


def get_kaggle_username():
    try:
        out = run_cmd("kaggle config view", retries=1)
        for line in out.split('\n'):
            if "username" in line.lower():
                return line.split(':')[1].strip()
    except Exception:
        print("[Kaggle] Make sure you have installed the kaggle CLI ('pip install kaggle') and placed your kaggle.json in ~/.kaggle/kaggle.json")
        sys.exit(1)
    return None

def is_bootstrap():
    return not os.path.exists(CHAIN_MARKER)


def setup_dataset(username, data_dir):
    dataset_slug = f"{username}/{DATASET_NAME}"
    os.makedirs(data_dir, exist_ok=True)

    metadata = {
        "title": "Polyfish Training Data",
        "id": dataset_slug,
        "licenses": [{"name": "CC0-1.0"}]
    }

    with open(os.path.join(data_dir, "dataset-metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"[Kaggle] Dataset metadata created for {dataset_slug}")
    return dataset_slug


def sync_dataset(username, bootstrap):
    dataset_slug = f"{username}/{DATASET_NAME}"
    os.makedirs(DATA_DIR, exist_ok=True)

    # Clean payload: everything except dataset-metadata.json is rebuilt
    for f in glob.glob(os.path.join(DATA_DIR, "*")):
        if os.path.basename(f) == "dataset-metadata.json":
            continue
        if os.path.isdir(f):
            shutil.rmtree(f)
        else:
            os.remove(f)

    shutil.copy2("train.py", DATA_DIR)
    import zipfile
    
    def stage_as_zip(src_path, dest_name):
        zip_path = os.path.join(DATA_DIR, dest_name + ".zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            zf.write(src_path, arcname=dest_name)
            
    pending = sorted(glob.glob(os.path.join(PENDING_DIR, "games_*.safetensors")))
    for f in pending:
        stage_as_zip(f, os.path.basename(f))
    print(f"[Kaggle] Staged and compressed {len(pending)} pending game file(s) for upload")

    if bootstrap:
        print("[Kaggle] Bootstrap round: including local model and teacher games in the dataset")
        if os.path.exists("model.safetensors"):
            shutil.copy2("model.safetensors", DATA_DIR)
        for f in glob.glob("teachers/games_*.safetensors"):
            stage_as_zip(f, "teachers__" + os.path.basename(f))
        for f in glob.glob("archive/games_*.safetensors"):
            stage_as_zip(f, "archive__" + os.path.basename(f))

    metadata_path = os.path.join(DATA_DIR, "dataset-metadata.json")
    if not os.path.exists(metadata_path):
        setup_dataset(username, DATA_DIR)
        print("[Kaggle] Creating new dataset on Kaggle...")
        run_cmd(f"kaggle datasets create -p {DATA_DIR}")
    else:
        print("[Kaggle] Uploading new version to Kaggle dataset...")
        # Ignore failure here if it's because the dataset is unchanged (Kaggle throws 400 Bad Request)
        out = run_cmd(f"kaggle datasets version -p {DATA_DIR} -m 'Update training data' -d", fatal=False)
        if out is None:
            print("[Kaggle] Dataset version upload failed (likely unchanged/already uploaded). Proceeding...")

    print("[Kaggle] Waiting for dataset to be processed by Kaggle...")
    deadline = time.time() + DATASET_READY_TIMEOUT
    while time.time() < deadline:
        status = run_cmd(f"kaggle datasets status {dataset_slug}", fatal=False) or ""
        if "ready" in status.lower():
            print("[Kaggle] Dataset ready; buffer time for the new version to propagate...")
            time.sleep(30)
            return
        print("[Kaggle] Dataset still processing, waiting 10s...")
        time.sleep(10)
    print(f"[Kaggle] Dataset not ready after {DATASET_READY_TIMEOUT}s")
    sys.exit(1)


# __BOOTSTRAP__ / __WINDOW__ / __ENV_OVERRIDES__ are substituted at push time.
SETUP_CELL = """\
import os, glob, shutil
import torch
print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())

def find_input(sentinel):
    for root, dirs, files in os.walk('/kaggle/input'):
        if sentinel in files:
            return root
    return None

wd = '/kaggle/working'
os.chdir(wd)
dataset_path = find_input('train.py')
if not dataset_path:
    raise FileNotFoundError('dataset with train.py not found under /kaggle/input')
print('dataset:', dataset_path)
prev_out = find_input('chain_state.json')
print('previous output:', prev_out)
os.makedirs('archive', exist_ok=True)
os.makedirs('teachers', exist_ok=True)
for item in os.listdir(dataset_path):
    s = os.path.join(dataset_path, item)
    if os.path.isdir(s):
        shutil.copytree(s, os.path.join(wd, item), dirs_exist_ok=True)
    elif item.startswith('teachers__'):
        shutil.copy2(s, os.path.join('teachers', item[len('teachers__'):]))
    elif item.startswith('archive__'):
        shutil.copy2(s, os.path.join('archive', item[len('archive__'):]))
    else:
        shutil.copy2(s, os.path.join(wd, item))
BOOTSTRAP = __BOOTSTRAP__
if prev_out and not BOOTSTRAP:
    for f in glob.glob(os.path.join(prev_out, 'archive', 'games_*.safetensors')):
        shutil.copy2(f, 'archive')
    for f in glob.glob(os.path.join(prev_out, 'teachers', 'games_*.safetensors')):
        shutil.copy2(f, 'teachers')
    prev_model = os.path.join(prev_out, 'model.safetensors')
    if os.path.exists(prev_model):
        shutil.copy2(prev_model, 'model.safetensors')
print('fresh games:', len(glob.glob('games_*.safetensors')),
      '| replay window:', len(glob.glob('archive/games_*.safetensors')),
      '| model present:', os.path.exists('model.safetensors'))
"""

TRAIN_CELL = """\
import os, re, glob, json, shutil, subprocess
os.chdir('/kaggle/working')
env = dict(os.environ)
env.update(__ENV_OVERRIDES__)
subprocess.check_call(['python', 'train.py'], env=env)
if os.path.exists('.last_train_metrics.json'):
    shutil.copy2('.last_train_metrics.json', 'last_train_metrics.json')
# Consolidate the replay window into archive/ for the next chained run
for f in glob.glob('games_*.safetensors'):
    shutil.move(f, os.path.join('archive', os.path.basename(f)))

def game_key(p):
    m = re.search(r'games_(\\d+)', os.path.basename(p))
    return (int(m.group(1)) if m else 0, os.path.basename(p))

window = sorted(glob.glob('archive/games_*.safetensors'), key=game_key)
for f in window[:-__WINDOW__]:
    os.remove(f)
# train.py must not linger in the output: its presence is how the setup cell
# tells the dataset mount apart from this kernel's own previous output.
os.remove('train.py')
with open('chain_state.json', 'w') as f:
    json.dump({'replay_files': min(len(window), __WINDOW__)}, f)
print('chain output ready:', min(len(window), __WINDOW__), 'replay file(s)')
"""


def kernel_env_overrides():
    env = {
        "TRAIN_EPOCHS": os.environ.get("TRAIN_EPOCHS", "5"),
        "TRAIN_LR": os.environ.get("TRAIN_LR", "0.001"),
        "REPLAY_BUFFER_FILES": str(REPLAY_WINDOW),
    }
    for k in ("DETACH_VALUE_TRUNK", "VALUE_LOSS_WEIGHT", "OWNERSHIP_LOSS_WEIGHT"):
        if k in os.environ:
            env[k] = os.environ[k]
    return env

def setup_kernel(username, dataset_slug, bootstrap):
    kernel_slug = f"{username}/{KERNEL_NAME}"
    os.makedirs(KERNEL_DIR, exist_ok=True)

    metadata = {
        "id": kernel_slug,
        "title": "Polyfish Trainer",
        "code_file": "train_notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [dataset_slug],
        "competition_sources": [],
        # Self-reference: chained runs read the previous run's output (model +
        # replay window). Bootstrap runs attach nothing so stale outputs from
        # the pre-chain notebook can never be picked up.
        "kernel_sources": [] if bootstrap else [kernel_slug],
    }

    with open(os.path.join(KERNEL_DIR, "kernel-metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    setup_src = SETUP_CELL.replace("__BOOTSTRAP__", str(bool(bootstrap)))
    train_src = (
        TRAIN_CELL
        .replace("__ENV_OVERRIDES__", json.dumps(kernel_env_overrides()))
        .replace("__WINDOW__", str(REPLAY_WINDOW))
    )

    def code_cell(src):
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": src.splitlines(keepends=True),
        }

    notebook = {
        "cells": [code_cell(setup_src), code_cell(train_src)],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
            "kaggle": {
                "accelerator": "nvidiaTeslaT4",
                "language": "python",
                "kernelType": "notebook",
                "isGpuEnabled": True
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    with open(os.path.join(KERNEL_DIR, "train_notebook.ipynb"), "w") as f:
        json.dump(notebook, f, indent=4)

    print(f"[Kaggle] Kernel metadata created (bootstrap={bootstrap})")
    return KERNEL_DIR, kernel_slug


def run_training(username, bootstrap):
    dataset_slug = f"{username}/{DATASET_NAME}"
    kernel_dir, kernel_slug = setup_kernel(username, dataset_slug, bootstrap)

    print(f"[Kaggle] Pushing and running kernel...")
    run_cmd(f"kaggle kernels push -p {kernel_dir} --accelerator NvidiaTeslaT4")

    print("[Kaggle] Waiting for training to complete...")
    deadline = time.time() + KERNEL_TIMEOUT
    while time.time() < deadline:
        time.sleep(30)
        status = run_cmd(f"kaggle kernels status {kernel_slug}", retries=1, fatal=False) or ""
        #print(f"Status: {status.strip()}")
        s = status.lower()
        if "complete" in s:
            print("[Kaggle] Training completed!")
            return
        if "error" in s or "cancel" in s:
            print("[Kaggle] Training failed!")
            sys.exit(1)
    print(f"[Kaggle] Kernel not complete after {KERNEL_TIMEOUT}s")
    sys.exit(1)


def pull_results(username):
    kernel_slug = f"{username}/{KERNEL_NAME}"
    print("[Kaggle] Downloading trained model and metrics...")
    shutil.rmtree("./kaggle_output", ignore_errors=True)

    # Patterned download avoids pulling the whole replay window back; fall
    # back to a full download if the installed CLI lacks --file-pattern.
    for pattern in ("model.safetensors", "last_train_metrics.json"):
        run_cmd(
            f"kaggle kernels output {kernel_slug} -p ./kaggle_output --file-pattern {pattern}",
            retries=1,
            fatal=False,
        )
    if not os.path.exists("./kaggle_output/model.safetensors"):
        print("[Kaggle] Patterned download failed; downloading full kernel output...")
        shutil.rmtree("./kaggle_output", ignore_errors=True)
        run_cmd(f"kaggle kernels output {kernel_slug} -p ./kaggle_output")

    if os.path.exists("./kaggle_output/model.safetensors"):
        shutil.copy2("./kaggle_output/model.safetensors", "./model.safetensors")
        print("[Kaggle] Updated local model.safetensors successfully!")
    else:
        print("[Kaggle] Error: model.safetensors not found in Kaggle output.")
        sys.exit(1)

    if os.path.exists("./kaggle_output/last_train_metrics.json"):
        shutil.copy2("./kaggle_output/last_train_metrics.json", ".last_train_metrics.json")
        print("[Kaggle] Pulled training metrics for training_log.")
    else:
        print("[Kaggle] Warning: last_train_metrics.json not found in Kaggle output; local metrics may be stale.")


def finalize_round():
    with open(CHAIN_MARKER, "w") as f:
        f.write(str(int(time.time())))
    removed = 0
    for f in glob.glob(os.path.join(PENDING_DIR, "games_*.safetensors")):
        os.remove(f)
        removed += 1
    print("[Kaggle] Round complete: chain established, {} pending file(s) cleared".format(removed))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("[Kaggle] Usage: python kaggle_manager.py [sync|train|pull|all]")
        sys.exit(1)

    action = sys.argv[1].lower()
    username = get_kaggle_username()

    if not username:
        print("[Kaggle] Could not retrieve username. Ensure kaggle config is set up.")
        sys.exit(1)

    print(f"[Kaggle] Using Kaggle username: {username}")
    bootstrap = is_bootstrap()

    if action == "sync":
        sync_dataset(username, bootstrap)
    elif action == "train":
        run_training(username, bootstrap)
    elif action == "pull":
        pull_results(username)
    elif action == "all":
        def fmt_time(t_secs):
            m, s = divmod(t_secs, 60)
            h, m = divmod(m, 60)
            if h > 0: return f"{int(h)}h {int(m)}m {int(s)}s"
            return f"{int(m)}m {int(s)}s"

        print("[Kaggle] --- Starting Full Kaggle Pipeline ---")
        t_start = time.time()
        
        print("[Kaggle] Syncing Dataset...")
        t_sync_start = time.time()
        sync_dataset(username, bootstrap)
        t_sync_end = time.time()
        print(f"[Kaggle] ✅ Sync Dataset finished in {fmt_time(t_sync_end - t_sync_start)}.")
        
        print("[Kaggle] Training...")
        t_train_start = time.time()
        run_training(username, bootstrap)
        t_train_end = time.time()
        print(f"[Kaggle] ✅ Training finished in {fmt_time(t_train_end - t_train_start)}.")
        
        print("[Kaggle] Pulling Results...")
        t_pull_start = time.time()
        pull_results(username)
        t_pull_end = time.time()
        print(f"[Kaggle] ✅ Pull Results finished in {fmt_time(t_pull_end - t_pull_start)}.")
        
        finalize_round()
        t_end = time.time()
        print(f"[Kaggle] 🎉 Pipeline Complete! Took: {fmt_time(t_end - t_start)}")
    else:
        print(f"[Kaggle] Unknown action: {action}")
        sys.exit(1)
