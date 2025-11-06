#!/bin/bash

# Slurm setup
#SBATCH -p gu-compute
#SBATCH -A gu-account
#SBATCH --qos=gu-med
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

echo "==================================="
echo "min_mdlm run"
date
echo "Job running on node: $(hostname)"
echo "==================================="

mkdir -p "../logs"

DATE_WITH_TIME=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="../logs/${DATE_WITH_TIME}_run_${SLURM_JOB_ID}.log"

echo "Attempting to activate venv"
source ~/miniconda3/bin/activate min_mdlm

echo "[DEBUG] Conda env check:"
echo "CONDA_DEFAULT_ENV: $CONDA_DEFAULT_ENV"
echo "[DEBUG] Python check:"
python --version
echo "==================================="

PROJECT_DIR="../"
echo "Attempting to change directory to: $PROJECT_DIR"
cd $PROJECT_DIR || { echo "ERROR: Could not find $PROJECT_DIR. Exiting."; exit 1; }

echo "[DEBUG] Current directory check:"
pwd

python -m main >> "$OUTPUT_FILE" 2>&1

echo "==================================="
echo "Fin."
date