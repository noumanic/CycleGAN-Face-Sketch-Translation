#!/usr/bin/env bash
set -e
echo "============================================"
echo " CycleGAN Face <-> Sketch  |  Local GPU"
echo "============================================"
echo

# Activate your environment if needed
# source .venv/bin/activate
# conda activate cyclegan

RESUME_FLAG=""
if [ -f "checkpoints/latest.pth" ]; then
    echo "[Launcher] Resuming from checkpoints/latest.pth"
    RESUME_FLAG="--resume"
else
    echo "[Launcher] No checkpoint found – starting fresh"
fi

echo "[Launcher] Starting training..."
echo

python local_train.py $RESUME_FLAG "$@"

echo
echo "[Launcher] Done. Checkpoint: checkpoints/latest.pth"