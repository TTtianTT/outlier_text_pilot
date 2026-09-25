#!/usr/bin/env bash
set -euo pipefail
python -m otc all --config configs/smoke.yaml --run runs/smoke
python -m otc verify-readout --run runs/smoke --limit 50
