#!/usr/bin/env bash
set -euo pipefail
python -m otc all --config configs/pilot.yaml --run runs/pilot
python -m otc verify-readout --run runs/pilot --limit 100
python -m otc plot --run runs/pilot
