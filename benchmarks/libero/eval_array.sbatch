#!/bin/bash
#SBATCH --job-name=vlash-libero-eval
#SBATCH --partition=8gpus
#SBATCH --account=MST114563
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --array=0-11%8
#SBATCH --output=/work/roboleon1295/vlash-eval/eval-%A_%a.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leon129506@gmail.com

set -x
SUITES=(libero_spatial libero_object libero_goal libero_10)
DELAYS=(0 1 3)
SUITE=${SUITES[$((SLURM_ARRAY_TASK_ID / 3))]}
DELAY=${DELAYS[$((SLURM_ARRAY_TASK_ID % 3))]}
PORT=$((5901 + SLURM_ARRAY_TASK_ID))
CKPT=/work/roboleon1295/vlash/outputs/train/async_libero/checkpoints/030000/pretrained_model
OUT=/work/roboleon1295/vlash-eval/results
mkdir -p $OUT

module purge; module load miniconda3/26.1.1
source activate /work/roboleon1295/envs/vlash
export LD_LIBRARY_PATH=/work/roboleon1295/envs/vlash/lib:${LD_LIBRARY_PATH:-}
export HOME=/work/roboleon1295/jobhome HF_HOME=/work/roboleon1295/.cache/huggingface
cd /work/roboleon1295/vlash

python benchmarks/libero/serve_policy.py --checkpoint $CKPT --port $PORT \
  > /work/roboleon1295/vlash-eval/server-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log 2>&1 &
SERVER_PID=$!
until curl -s http://127.0.0.1:$PORT/health | grep -q ok; do
  kill -0 $SERVER_PID || { echo "SERVER DIED"; exit 1; }
  sleep 5
done

for attempt in 1 2 3 4 5; do
  MUJOCO_GL=egl /work/roboleon1295/.venv-libero-vanilla/bin/python \
    benchmarks/libero/eval_client.py --suite $SUITE --delay $DELAY --port $PORT \
    --out $OUT/${SUITE}_d${DELAY}.json
  RC=$?
  [ $RC -eq 0 ] && break
  echo "CLIENT CRASHED (rc=$RC), retry $attempt"
  sleep 10
done
kill $SERVER_PID
[ $RC -ne 0 ] && exit 1
