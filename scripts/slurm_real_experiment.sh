#!/usr/bin/env bash
#SBATCH --job-name=otc-real
#SBATCH --partition=RTXq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=runs/slurm-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PATH="$PWD/.venv/bin:$PATH"
export HF_HOME="$PWD/runs/hf_cache"
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
mkdir -p runs/real_20260925
python -m pip freeze > runs/real_20260925/environment.txt
nvidia-smi
python -m unittest discover -s tests -v
python -m otc generate --config configs/smoke.yaml --run runs/real_20260925/smoke
python -m otc features --config configs/smoke.yaml --run runs/real_20260925/smoke
if python -m otc evaluate --run runs/real_20260925/smoke; then
    python -m otc verify-readout --run runs/real_20260925/smoke --limit 50 || {
        status=$?
        if [[ "$status" != 3 ]]; then exit "$status"; fi
        echo 'Smoke readout did not pass; status retained in readout_verification.json.'
    }
else
    echo 'Smoke evaluation failed; inspect log. Full pilot retains original protocol.'
fi
python -m otc all --config configs/pilot.yaml --run runs/real_20260925/pilot
python -m otc verify-readout --run runs/real_20260925/pilot --limit 100 || {
    status=$?
    if [[ "$status" != 3 ]]; then exit "$status"; fi
    echo 'Pilot readout did not pass; status retained in readout_verification.json.'
}
python -m otc plot --run runs/real_20260925/pilot
python -m otc codec-bench --run runs/real_20260925/pilot --out runs/real_20260925/pilot_codec_1byte --trials 20 --bytes 1 --bits 1 --margin 0.0001
echo 'EXPERIMENT_COMPLETE'
