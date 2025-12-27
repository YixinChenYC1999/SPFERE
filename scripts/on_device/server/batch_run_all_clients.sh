#!/bin/bash
# Batch run client_test.sh on all clients and live-tail their logs
# Location: scripts/on_device/server/batch_run_all_clients.sh

set -euo pipefail

# -----------------------------------------------------
# 0. Resolve project root (three levels up)
# scripts/on_device/server -> ../../.. -> project_root
# -----------------------------------------------------
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

echo "=== SPFERE Batch Run All Clients ==="
echo "Project root: $ROOT_DIR"
echo

# -----------------------------------------------------
# 1. Clients alias and remote paths
# -----------------------------------------------------
CLIENTS=(
  "raspi_4_1"
  "raspi_4_2"
  "raspi_4_3"
  "raspi_4_4"
  "raspi_5_1"
  "raspi_5_2"
  "raspi_5_3"
  "raspi_5_4"
  "opi5_plus"
  "rockpi4c+"
)

# Client-side project root: ~/tmc
REMOTE_DIR="pi_comu/tmc"
# New location of client_test.sh inside client project
REMOTE_SCRIPT="scripts/on_device/client/client_test.sh"
REMOTE_LOG="client_log/log.log"
REMOTE_DEBUG_LOG="client_log/debug.log"

# Whether to run clients in debug mode (passed as $1 to client_test.sh)
DEBUG_ENABLED=false   # set to true if you want debug logs

mkdir -p server_log

# -----------------------------------------------------
# 2. Start client_test.sh on all clients
# -----------------------------------------------------
for CLIENT in "${CLIENTS[@]}"; do
  echo "Starting client_test.sh on $CLIENT... (debug=$DEBUG_ENABLED)"
  ssh "$CLIENT" "
    cd ~/$REMOTE_DIR || { echo '[ERROR] REMOTE_DIR not found on $CLIENT'; exit 1; }
    bash $REMOTE_SCRIPT $DEBUG_ENABLED
  " &
done

# 给客户端一点时间启动
sleep 10

# -----------------------------------------------------
# 3. Tail logs from all clients
# -----------------------------------------------------
for CLIENT in "${CLIENTS[@]}"; do
  echo "Tailing logs from $CLIENT..."

  # Tail main log
  ssh "$CLIENT" "
    cd ~/$REMOTE_DIR || { echo '[ERROR] REMOTE_DIR not found on $CLIENT'; exit 1; }
    tail -n 100 -f $REMOTE_LOG
  " | tee "server_log/za_output_${CLIENT}.log" &

  sleep 1

  # If debug enabled, also tail debug.log
  if [[ "$DEBUG_ENABLED" == true ]]; then
    ssh "$CLIENT" "
      cd ~/$REMOTE_DIR || { echo '[ERROR] REMOTE_DIR not found on $CLIENT'; exit 1; }
      tail -n 100 -f $REMOTE_DEBUG_LOG
    " | tee "server_log/zb_debug_${CLIENT}.log" &

    sleep 1
  fi
done

echo "All clients started. Live tailing logs now. Press Ctrl+C to stop."
wait
