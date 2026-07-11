import os
import sys
import json
import time
import subprocess
from pathlib import Path

# Ensure .venv/bin is in PATH so the kaggle CLI is found
os.environ["PATH"] = f"{os.path.abspath('.venv/bin')}:{os.environ['PATH']}"

# ==========================================
# Kaggle Integration Manager for Polyfish
# ==========================================
# This script automates sending your generated games to Kaggle,
# training the model there using their free GPUs (T4x2 or P100),
# and pulling the newly trained model back.

DATASET_NAME = "polyfish-training-data"
KERNEL_NAME = "polyfish-trainer"

def run_cmd(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if result.returncode != 0:
        print(f"Error executing {cmd}:")
        print(result.stderr)
        sys.exit(1)
    return result.stdout.strip()

def get_kaggle_username():
    try:
        out = run_cmd("kaggle config view")
        for line in out.split('\n'):
            if "username" in line.lower():
                return line.split(':')[1].strip()
    except Exception:
        print("Make sure you have installed the kaggle CLI ('pip install kaggle') and placed your kaggle.json in ~/.kaggle/kaggle.json")
        sys.exit(1)
    return None

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
        
    print(f"Dataset metadata created for {dataset_slug}")
    return dataset_slug

def sync_dataset(username):
    dataset_slug = f"{username}/{DATASET_NAME}"
    data_dir = "kaggle_data_sync"
    
    print("Preparing data for upload...")
    os.makedirs(data_dir, exist_ok=True)
    
    # Clear old games to prevent accumulation
    run_cmd(f"rm -f {data_dir}/games_*.safetensors")
    
    # Copy necessary training files
    run_cmd(f"cp train.py {data_dir}/")
    if os.path.exists("model.safetensors"):
        run_cmd(f"cp model.safetensors {data_dir}/")
    
    if os.path.exists("archive"):
        run_cmd(f"cp archive/games_*.safetensors {data_dir}/ 2>/dev/null || true")
    if os.path.exists("teachers"):
        run_cmd(f"cp teachers/games_*.safetensors {data_dir}/ 2>/dev/null || true")
        
    # Copy fresh games if they exist
    run_cmd(f"cp games_*.safetensors {data_dir}/ 2>/dev/null || true")
    
    metadata_path = os.path.join(data_dir, "dataset-metadata.json")
    if not os.path.exists(metadata_path):
        setup_dataset(username, data_dir)
        print("Creating new dataset on Kaggle...")
        run_cmd(f"kaggle datasets create -p {data_dir}")
    else:
        print("Uploading new version to Kaggle dataset...")
        run_cmd(f"kaggle datasets version -p {data_dir} -m 'Update training data' -d")

    print("Waiting for dataset to be processed by Kaggle...")
    import time
    while True:
        status = run_cmd(f"kaggle datasets status {dataset_slug}")
        if "ready" in status.lower():
            print("Dataset reports ready, adding buffer time for version to propagate...")
            time.sleep(30) # Wait for the new version to actually be available
            break
        print("Dataset still processing, waiting 10s...")
        time.sleep(10)

def setup_kernel(username, dataset_slug):
    kernel_slug = f"{username}/{KERNEL_NAME}"
    kernel_dir = "kaggle_kernel_sync"
    os.makedirs(kernel_dir, exist_ok=True)
    
    metadata = {
      "id": kernel_slug,
      "title": "Polyfish Trainer",
      "code_file": "train_notebook.ipynb",
      "language": "python",
      "kernel_type": "notebook",
      "is_private": "true",
      "enable_gpu": "true",
      "enable_internet": "true",
      "dataset_sources": [dataset_slug],
      "competition_sources": [],
      "kernel_sources": []
    }
    
    with open(os.path.join(kernel_dir, "kernel-metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
        
    # Create the Jupyter Notebook
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "!pip install safetensors torch\n",
                    f"import os\n",
                    f"import shutil\n",
                    f"dataset_path = None\n",
                    f"for root, dirs, files in os.walk('/kaggle/input'):\n",
                    f"    if 'train.py' in files:\n",
                    f"        dataset_path = root\n",
                    f"        break\n",
                    f"if not dataset_path:\n",
                    f"    raise FileNotFoundError('Could not find train.py in /kaggle/input')\n",
                    f"print('Found dataset path:', dataset_path)\n",
                    f"working_dir = '/kaggle/working'\n",
                    f"# Copy dataset to working dir so train.py can write/read properly\n",
                    f"for item in os.listdir(dataset_path):\n",
                    f"    s = os.path.join(dataset_path, item)\n",
                    f"    d = os.path.join(working_dir, item)\n",
                    f"    if os.path.isdir(s):\n",
                    f"        shutil.copytree(s, d, dirs_exist_ok=True)\n",
                    f"    else:\n",
                    f"        shutil.copy2(s, d)\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os\n",
                    "import subprocess\n",
                    "os.environ['TRAIN_EPOCHS'] = '5'\n", # Override env vars here if needed
                    "os.environ['TRAIN_LR'] = '0.001'\n",
                    "subprocess.check_call(['python', 'train.py'])\n",
                    "subprocess.call('rm -rf games_*.safetensors archive', shell=True)\n"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open(os.path.join(kernel_dir, "train_notebook.ipynb"), "w") as f:
        json.dump(notebook, f, indent=4)
        
    print(f"Kernel metadata created for {kernel_slug}")
    return kernel_dir, kernel_slug

def run_training(username):
    dataset_slug = f"{username}/{DATASET_NAME}"
    kernel_dir, kernel_slug = setup_kernel(username, dataset_slug)
    
    print(f"Pushing and running kernel {kernel_slug}...")
    run_cmd(f"kaggle kernels push -p {kernel_dir}")
    
    print("Waiting for training to complete... (This will take a while, checking every 60s)")
    while True:
        time.sleep(60)
        status_out = subprocess.run(f"kaggle kernels status {kernel_slug}", shell=True, text=True, capture_output=True).stdout
        print(f"Status: {status_out.strip()}")
        if "complete" in status_out.lower():
            print("Training completed!")
            break
        elif "error" in status_out.lower():
            print("Training failed!")
            sys.exit(1)

def pull_results(username):
    kernel_slug = f"{username}/{KERNEL_NAME}"
    print("Downloading trained model and metrics...")
    run_cmd("rm -rf ./kaggle_output")
    run_cmd(f"kaggle kernels output {kernel_slug} -p ./kaggle_output --file-pattern model.safetensors")
    
    # Move the new model to the polyfish-rs root
    if os.path.exists("./kaggle_output/model.safetensors"):
        run_cmd("cp ./kaggle_output/model.safetensors ./model.safetensors")
        print("Updated local model.safetensors successfully!")
    else:
        print("Warning: model.safetensors not found in Kaggle output.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python kaggle_manager.py [sync|train|pull|all]")
        sys.exit(1)
        
    action = sys.argv[1].lower()
    username = get_kaggle_username()
    
    if not username:
        print("Could not retrieve Kaggle username. Ensure kaggle config is set up.")
        sys.exit(1)
        
    print(f"Using Kaggle username: {username}")
    
    if action == "sync":
        sync_dataset(username)
    elif action == "train":
        run_training(username)
    elif action == "pull":
        pull_results(username)
    elif action == "all":
        sync_dataset(username)
        run_training(username)
        pull_results(username)
    else:
        print(f"Unknown action: {action}")
