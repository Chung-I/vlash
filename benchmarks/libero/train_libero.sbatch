#!/bin/bash
#SBATCH --job-name=vlash-libero-train
#SBATCH --partition=8gpus
#SBATCH --account=MST114563
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=800G
#SBATCH --time=12:00:00
#SBATCH --output=/work/roboleon1295/vlash-eval/train-%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leon129506@gmail.com

set -x
module purge; module load miniconda3/26.1.1
source activate /work/roboleon1295/envs/vlash
export LD_LIBRARY_PATH=/work/roboleon1295/envs/vlash/lib:${LD_LIBRARY_PATH:-}
export HOME=/work/roboleon1295/jobhome
mkdir -p "$HOME/.triton" "$HOME/.cache"
export TRITON_CACHE_DIR="$HOME/.triton" XDG_CACHE_HOME="$HOME/.cache"
export HF_HOME=/work/roboleon1295/.cache/huggingface
cd /work/roboleon1295/vlash

srun vlash train examples/train/pi05/async.yaml HuggingFaceVLA/libero \
  --output_dir=outputs/train/async_libero \
  --job_name=async_libero \
  --steps=30000 \
  --batch_size=8 \
  --max_delay_steps=3 \
  --use_state_ground_truth=true \
  --num_workers=8 \
  --save_freq=10000 \
  --scheduler.num_decay_steps=30000 \
  --wandb.project=vlash-libero
