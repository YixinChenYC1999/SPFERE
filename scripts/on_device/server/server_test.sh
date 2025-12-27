#!/bin/bash
# Server test script using GNU screen.
# Supports:
#   ./server_test.sh         # normal
#   ./server_test.sh true    # debug mode (log to server_log/debug.log)

set -euo pipefail

# -----------------------------------------------------
# 0. Resolve project root (this script is 3 levels deep)
# scripts/on_device/server -> ../../.. -> project root
# -----------------------------------------------------
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

# -----------------------------------------------------
# 1. Activate venv
# -----------------------------------------------------
if [[ ! -d server_venv ]]; then
    echo "ERROR: server_venv/ not found. run server_setup.sh first."
    exit 1
fi

# shellcheck disable=SC1091
source server_venv/bin/activate

# -----------------------------------------------------
# 2. Screen / Debug config
# -----------------------------------------------------
SCREEN_NAME="server_test"
DEBUG_FLAG=false

if [[ $# -ge 1 ]]; then
    DEBUG_FLAG="$1"
fi

# -----------------------------------------------------
# 3. Kill old screen if exists
# -----------------------------------------------------
if screen -list | grep -q "$SCREEN_NAME"; then
    echo "Old screen session $SCREEN_NAME detected. Killing it..."
    screen -S "$SCREEN_NAME" -X quit || true
    sleep 1
fi

# -----------------------------------------------------
# 4. Build command for inside screen
# -----------------------------------------------------
if [[ "$DEBUG_FLAG" == true ]]; then
    SCREEN_CMD='
        ts=$(date +%Y%m%d_%H%M%S)

        # Rotate output dirs
        [ -d "server_model" ] && mv server_model "server_model_before_${ts}"
        mkdir -p server_model

        [ -d "server_log" ] && mv server_log "server_log_before_${ts}"
        mkdir -p server_log

        echo "[DEBUG] Server started at $(date)" >> server_log/debug.log
        python -m chen2026spfere.server.server 2>&1 | tee -a server_log/debug.log
        echo "[DEBUG] Server finished at $(date)" >> server_log/debug.log

        exec bash
    '
else
    SCREEN_CMD='
        ts=$(date +%Y%m%d_%H%M%S)

        [ -d "server_model" ] && mv server_model "server_model_before_${ts}"
        mkdir -p server_model

        [ -d "server_log" ] && mv server_log "server_log_before_${ts}"
        mkdir -p server_log

        python -m chen2026spfere.server.server

        exec bash
    '
fi

# -----------------------------------------------------
# 5. Start screen and attach
# -----------------------------------------------------
screen -dmS "$SCREEN_NAME" bash -c "$SCREEN_CMD"

sleep 2
screen -r "$SCREEN_NAME"
