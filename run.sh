#!/bin/bash
#SBATCH --partition=aa100
#SBATCH --qos=gpu-testing
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/projects/%u/Multilingual-Stories-Topic-Space/logs/%j.log
#SBATCH --job-name=topic-space
#SBATCH --account=ucb-general
#SBATCH --mail-type=END,FAIL

SLURM_TMPDIR="${SLURM_SCRATCH:-/tmp/$USER/scratch}"
export HF_HOME="/scratch/alpine/$USER/.cache/huggingface"
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $HF_HOME

export LOCAL_OUTPUT_DIR="$SLURM_TMPDIR/outputs"

module load uv
module load python

uv sync
uv run run.py
