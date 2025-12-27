#!/bin/bash
# Sim runner using GNU screen.

set -euo pipefail

# -----------------------------------------------------
# 0. Resolve project root (two levels up)
# scripts/simulation -> ../.. -> project_root
# -----------------------------------------------------
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# 1) Activate venv
source sim_venv/bin/activate

# 2) Unique screen name
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SCREEN_NAME="sim_run_${TIMESTAMP}"

# 3) Generate command
SCREEN_CMD='


    python -m chen2026spfere.simulation.sim


    exec bash
'

# 4) Start detached, print name
screen -dmS "$SCREEN_NAME" bash -c "$SCREEN_CMD"
echo "Started screen session: $SCREEN_NAME"
sleep 3
screen -r "$SCREEN_NAME"