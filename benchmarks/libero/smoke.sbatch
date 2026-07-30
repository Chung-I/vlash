#!/bin/bash
#SBATCH --job-name=libero-smoke
#SBATCH --partition=dev
#SBATCH --account=MST114563
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=00:40:00
#SBATCH --output=/work/roboleon1295/vlash-eval/smoke-%j.log

set -x
mkdir -p /work/roboleon1295/vlash-eval
module purge; module load miniconda3/26.1.1
source activate /work/roboleon1295/envs/vlash
export LD_LIBRARY_PATH=/work/roboleon1295/envs/vlash/lib:${LD_LIBRARY_PATH:-}
export HOME=/work/roboleon1295/jobhome HF_HOME=/work/roboleon1295/.cache/huggingface
cd /work/roboleon1295/vlash

python benchmarks/libero/smoke_server.py > /work/roboleon1295/vlash-eval/smoke-server-$SLURM_JOB_ID.log 2>&1 &
SERVER_PID=$!
until curl -s http://127.0.0.1:5901/health | grep -q ok; do sleep 5; done

MUJOCO_GL=egl /work/roboleon1295/.venv-libero-vanilla/bin/python \
  benchmarks/libero/eval_client.py --suite libero_spatial --delay 1 \
  --num-trials 2 --out /work/roboleon1295/vlash-eval/smoke.json
kill $SERVER_PID
cat /work/roboleon1295/vlash-eval/smoke.json
