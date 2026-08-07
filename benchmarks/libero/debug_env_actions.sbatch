#!/bin/bash
#SBATCH --job-name=libero-env-action-diff
#SBATCH --partition=dev
#SBATCH --account=MST114563
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=00:40:00
#SBATCH --output=/work/roboleon1295/vlash-eval/env-action-diff-%j.log

# Env-action diff probe: the offline predict-vs-dataset probe showed
# lerobot/pi05_libero produces sane actions on real dataset frames
# (MAE 0.03-0.08, same league as our fine-tuned checkpoint), yet the
# live-env eval shard still scored 0%. This job runs ONE task, ONE
# episode, through the *actual* eval_client -> DelayedChunkExecutor ->
# server /predict path for both checkpoints back to back, logging the
# first 15 raw action vectors sent to env.step, to see whether served
# actions in the live env are frozen/near-zero, wrongly scaled, sign-
# flipped, or plausible-but-failing (visual domain gap).

set -x
mkdir -p /work/roboleon1295/vlash-eval
module purge; module load miniconda3/26.1.1
source activate /work/roboleon1295/envs/vlash
export LD_LIBRARY_PATH=/work/roboleon1295/envs/vlash/lib:${LD_LIBRARY_PATH:-}
export HOME=/work/roboleon1295/jobhome HF_HOME=/work/roboleon1295/.cache/huggingface
cd /work/roboleon1295/vlash

PORT=5901
OUT=/work/roboleon1295/vlash-eval/debug_env_actions

echo "===PI05_LIBERO==="
python benchmarks/libero/serve_policy.py \
  --checkpoint lerobot/pi05_libero --port $PORT \
  --stats-dataset HuggingFaceVLA/libero \
  > /work/roboleon1295/vlash-eval/env-action-diff-server-pi05_libero-$SLURM_JOB_ID.log 2>&1 &
SERVER_PID=$!
until curl -s http://127.0.0.1:$PORT/health | grep -q ok; do
  kill -0 $SERVER_PID || { echo "PI05_LIBERO SERVER DIED"; exit 1; }
  sleep 5
done
echo "--- pi05_libero server log (injection evidence) ---"
cat /work/roboleon1295/vlash-eval/env-action-diff-server-pi05_libero-$SLURM_JOB_ID.log

MUJOCO_GL=egl /work/roboleon1295/.venv-libero-vanilla/bin/python \
  benchmarks/libero/eval_client.py --suite libero_spatial --delay 0 --port $PORT \
  --num-trials 1 --task-id 0 --log-actions \
  --out ${OUT}_pi05_libero.json
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null

echo "===FINETUNED==="
python benchmarks/libero/serve_policy.py \
  --checkpoint /work/roboleon1295/vlash/outputs/train/async_libero/checkpoints/030000/pretrained_model \
  --port $PORT \
  > /work/roboleon1295/vlash-eval/env-action-diff-server-finetuned-$SLURM_JOB_ID.log 2>&1 &
SERVER_PID=$!
until curl -s http://127.0.0.1:$PORT/health | grep -q ok; do
  kill -0 $SERVER_PID || { echo "FINETUNED SERVER DIED"; exit 1; }
  sleep 5
done

MUJOCO_GL=egl /work/roboleon1295/.venv-libero-vanilla/bin/python \
  benchmarks/libero/eval_client.py --suite libero_spatial --delay 0 --port $PORT \
  --num-trials 1 --task-id 0 --log-actions \
  --out ${OUT}_finetuned.json
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null

echo "ENV_ACTION_DIFF_DONE"
