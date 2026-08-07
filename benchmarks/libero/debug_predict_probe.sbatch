#!/bin/bash
#SBATCH --job-name=libero-predict-probe
#SBATCH --partition=dev
#SBATCH --account=MST114563
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH --output=/work/roboleon1295/vlash-eval/predict-probe-%j.log

set -x
mkdir -p /work/roboleon1295/vlash-eval
module purge; module load miniconda3/26.1.1
source activate /work/roboleon1295/envs/vlash
export LD_LIBRARY_PATH=/work/roboleon1295/envs/vlash/lib:${LD_LIBRARY_PATH:-}
export HOME=/work/roboleon1295/jobhome HF_HOME=/work/roboleon1295/.cache/huggingface
cd /work/roboleon1295/vlash

python benchmarks/libero/debug_predict_probe.py
