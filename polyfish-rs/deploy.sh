#!/bin/bash
# Usage: ./deploy.sh
# You must set RUNPOD_HOST and RUNPOD_PORT env vars, or edit them below.

HOST="${RUNPOD_HOST:-}"
PORT="${RUNPOD_PORT:-}"
USER="root"

if [ -z "$HOST" ] || [ -z "$PORT" ]; then
    echo "Usage: export RUNPOD_HOST=x.x.x.x RUNPOD_PORT=xxxxx && ./deploy.sh"
    echo "Or edit this script to set defaults."
    read -p "Enter RunPod IP: " HOST
    read -p "Enter RunPod Port: " PORT
fi

echo "Deploying to $USER@$HOST:$PORT..."

# 1. Sync Files (Exclude artifacts, venv, target)
# We use rsync to push the current folder to /root/polyfish
rsync -avz -e "ssh -p $PORT" \
    --exclude 'target' \
    --exclude '.venv' \
    --exclude '*.safetensors' \
    --exclude '.git' \
    ./ $USER@$HOST:/root/polyfish/

# 2. Run Setup on Remote
echo "Running remote setup..."
ssh -p $PORT $USER@$HOST "cd /root/polyfish && chmod +x remote_setup.sh && ./remote_setup.sh"

echo "Deployment finished."
echo "To connect: ssh -p $PORT $USER@$HOST"
echo "To run training: ssh -p $PORT $USER@$HOST 'cd polyfish && ./run_training_loop.sh'"
