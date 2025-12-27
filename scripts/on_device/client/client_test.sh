#!/bin/bash
# Client test script with screen and optional debug log via argument

set -euo pipefail

# Resolve project root
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

# Activate venv (adjust path if your venv name/path changes)
source "$ROOT_DIR/client_venv/bin/activate"

SCREEN_NAME="client_test"
DEBUG_FLAG=false

if [ "$#" -ge 1 ]; then
    DEBUG_FLAG="$1"
fi

# Kill old screen session if exists
if screen -list | grep -q "$SCREEN_NAME"; then
    echo "Old screen session $SCREEN_NAME detected. Killing it..."
    screen -S "$SCREEN_NAME" -X quit
    sleep 1
fi

# Common body of the screen command (without DEBUG / non-DEBUG specific parts)
make_dirs_and_rotate='
    if [ -d "client_model" ]; then
        mv "client_model" "client_model_before_$(date +%Y%m%d_%H%M%S)"
    fi
    mkdir -p client_model

    if [ -d "client_log" ]; then
        mv "client_log" "client_log_before_$(date +%Y%m%d_%H%M%S)"
    fi
    mkdir -p client_log
'

if [ "$DEBUG_FLAG" = true ]; then
    SCREEN_CMD='
        '"$make_dirs_and_rotate"'
        echo "[DEBUG] Client started at $(date)" >> client_log/debug.log
        python -m chen2026spfere.client.client 2>&1 | tee -a client_log/debug.log
        echo "[DEBUG] Client finished at $(date)" >> client_log/debug.log
        exec bash
    '
else
    SCREEN_CMD='
        '"$make_dirs_and_rotate"'
        python -m chen2026spfere.client.client
        exec bash
    '
fi

screen -dmS "$SCREEN_NAME" bash -c "$SCREEN_CMD"

sleep 2
screen -r "$SCREEN_NAME"