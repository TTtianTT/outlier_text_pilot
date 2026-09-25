#!/usr/bin/env bash
#SBATCH --job-name=otc-pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=otc-%j.log
# Set the valid account and partition for YOUR cluster when submitting.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from project root}"
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m otc all --config configs/pilot.yaml --run runs/pilot
python -m otc verify-readout --run runs/pilot --limit 100
