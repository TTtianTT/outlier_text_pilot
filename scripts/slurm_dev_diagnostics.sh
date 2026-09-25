#!/usr/bin/env bash
#SBATCH --job-name=otc-dev-diag
#SBATCH --partition=RTXq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=runs/dev-diagnostics-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PATH="$PWD/.venv/bin:$PATH"
export HF_HOME="$PWD/runs/hf_cache"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
python -m otc.diagnostics --run runs/real_20260925/pilot --out runs/dev_diagnostics_20260925
python -m otc verify-readout --run runs/real_20260925/pilot --scope eligible --split dev --bits 1 --limit 10000 --out runs/dev_diagnostics_20260925/readout_verification.json
echo DIAGNOSTICS_COMPLETE
