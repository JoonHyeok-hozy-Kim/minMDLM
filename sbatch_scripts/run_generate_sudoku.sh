#!/bin/bash

# Slurm setup (No GPU required! 100/256 CPU in Parallel)
#SBATCH -p gu-compute
#SBATCH -A gu-account
#SBATCH --qos=gu-med
#SBATCH --cpus-per-task=100
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

mkdir -p "./logs"

DATE_WITH_TIME=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="./logs/${DATE_WITH_TIME}_run_${SLURM_JOB_ID}.log"

exec > "$OUTPUT_FILE" 2>&1

echo "==================================="
echo "min_mdlm run"
date
echo "Job running on node: $(hostname)"
echo "==================================="

echo "Attempting to activate venv"
source ./venv/bin/activate

echo "[DEBUG] Python check:"
python --version
echo "==================================="
echo "Running manage_sudoku_files.py starts"
date

python -m tests.sudoku.manage_sudoku_files

echo "==================================="
echo "Fin."
date