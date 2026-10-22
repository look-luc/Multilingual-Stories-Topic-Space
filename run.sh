#!/bin/bash
#SBATCH --partition=artxpro6000
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=7:30:00
#SBATCH --output=/projects/%u/spec-decoding-capstone/logs/%j.log
#SBATCH --job-name=specdec_l40
#SBATCH --account=ucb-general
#SBATCH --mail-type=END,FAIL

SLURM_TMPDIR="${SLURM_SCRATCH:-/tmp/$USER/scratch}"
export HF_HOME="/scratch/alpine/$USER/.cache/huggingface"
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $HF_HOME

# 1. Define local scratch directories for active logging and checkpoints
export LOCAL_OUTPUT_DIR="$SLURM_TMPDIR/outputs"

module load uv
uv sync
uv run python run.py
