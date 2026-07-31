#!/bin/bash
#SBATCH --job-name=libero10-microbench
#SBATCH --partition=dev
#SBATCH --account=MST114563
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=00:40:00
#SBATCH --output=/work/roboleon1295/stage-latency/libero10-microbench-%j.log

source /work/roboleon1295/.venv-libero-vanilla/bin/activate
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
cd /work/roboleon1295/vlash

SCRIPT=/work/roboleon1295/vlash/benchmarks/libero/debug_microbench_libero10.py

echo "=== nvidia-smi at start ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv

echo "=== STEP 1: baseline, exact shard kwargs, both suites ==="
python -u "$SCRIPT" --suite libero_spatial --n 50 --label baseline
python -u "$SCRIPT" --suite libero_10      --n 50 --label baseline

echo "=== STEP 2: libero_10 knob variants ==="
python -u "$SCRIPT" --suite libero_10 --n 50 --label omit_render_gpu_id --omit-render-gpu-id
python -u "$SCRIPT" --suite libero_10 --n 50 --label hard_reset_false --no-hard-reset
python -u "$SCRIPT" --suite libero_10 --n 50 --label cam128 --camera-size 128

echo "=== STEP 2d: MUJOCO_GL=osmesa comparison (diagnostic, separate process) ==="
MUJOCO_GL=osmesa python -u "$SCRIPT" --suite libero_10 --n 20 --label osmesa

echo "=== nvidia-smi at end ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv

echo "MICROBENCH_DONE"
