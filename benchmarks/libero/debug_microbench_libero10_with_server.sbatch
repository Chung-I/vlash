#!/bin/bash
#SBATCH --job-name=libero10-mb-srv
#SBATCH --partition=dev
#SBATCH --account=MST114563
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH --output=/work/roboleon1295/stage-latency/libero10-mb-srv-%j.log

set -e
CKPT=/work/roboleon1295/vlash/outputs/train/async_libero/checkpoints/030000/pretrained_model
PORT=5999

echo "=== starting pi05 server (conda env vlash) on a FRESH node/GPU, no crash history ==="
module purge; module load miniconda3/26.1.1
source activate /work/roboleon1295/envs/vlash
export LD_LIBRARY_PATH=/work/roboleon1295/envs/vlash/lib:${LD_LIBRARY_PATH:-}
export HOME=/work/roboleon1295/jobhome HF_HOME=/work/roboleon1295/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0
cd /work/roboleon1295/vlash
python -u benchmarks/libero/serve_policy.py --checkpoint "$CKPT" --port $PORT \
    > /work/roboleon1295/stage-latency/mb-srv-server-$SLURM_JOB_ID.log 2>&1 &
SERVER_PID=$!
echo "server pid $SERVER_PID"

echo "=== waiting for /health ==="
for i in $(seq 1 60); do
  if curl -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "server healthy after ${i}s"
    break
  fi
  sleep 1
done

echo "=== nvidia-smi with server loaded (idle, no traffic yet) ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv

echo "=== deactivate conda, switch to libero venv for the render micro-bench ==="
conda deactivate 2>/dev/null || true
source /work/roboleon1295/.venv-libero-vanilla/bin/activate
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
cd /work/roboleon1295/vlash

echo "=== STEP A: libero_10 render loop, server LOADED but IDLE (no /predict traffic) ==="
python -u benchmarks/libero/debug_microbench_libero10.py --suite libero_10 --n 50 --label with_idle_server

echo "=== STEP B: libero_10 render loop, WHILE actively hammering /predict concurrently ==="
python -u - <<PYEOF &
import io, time, numpy as np, requests
img = (np.random.rand(256, 256, 3) * 255).astype("uint8")
state = np.zeros(8, dtype="float32")
t_end = time.time() + 40
n = 0
while time.time() < t_end:
    buf = io.BytesIO()
    np.savez(buf, image=img, wrist_image=img, state=state, task=np.array("dummy task"))
    try:
        requests.post("http://127.0.0.1:$PORT/predict", data=buf.getvalue(), timeout=60)
        n += 1
    except Exception as e:
        print("predict error", e, flush=True)
print(f"hammer thread issued {n} predict calls", flush=True)
PYEOF
HAMMER_PID=$!
sleep 2
python -u benchmarks/libero/debug_microbench_libero10.py --suite libero_10 --n 50 --label with_active_server_load
wait $HAMMER_PID

echo "=== nvidia-smi at end ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv

kill $SERVER_PID 2>/dev/null || true
echo "MICROBENCH_WITH_SERVER_DONE"
