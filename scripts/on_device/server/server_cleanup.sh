#!/bin/bash
# Cleanup script for SPFERE server

set -euo pipefail

# -----------------------------------------------------
# 0. Resolve project root (three levels up)
# -----------------------------------------------------
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

echo "=== SPFERE Server Cleanup ==="
echo "Project root: $ROOT_DIR"
echo

# -----------------------------------------------------
# 1. First confirmation: delete basic logs + models?
# -----------------------------------------------------
if [[ $# -ge 1 ]]; then
    confirmlog="$1"
else
    read -p "Delete all basic outputs (server_log*, server_model*)? (y/n): " confirmlog
fi

if [[ "$confirmlog" =~ ^([yY]|[yY][eE][sS])$ ]]; then
    rm -rf server_model*
    rm -rf server_log*
    echo "[OK] Deleted server_model* and server_log*"
else
    echo "[Skip] Basic output not deleted."
fi

echo

# -----------------------------------------------------
# 2. Second confirmation: delete venv + data + __pycache__ ?
# -----------------------------------------------------
if [[ $# -ge 2 ]]; then
    confirmex="$2"
else
    read -p "Delete venv, data/, and __pycache__? (y/n): " confirmex
fi

if [[ "$confirmex" =~ ^([yY]|[yY][eE][sS])$ ]]; then
    rm -rf server_venv
    rm -rf data
    rm -rf src/chen2026spfere/**/__pycache__ 2>/dev/null || true
    echo "[OK] Deleted server_venv, data/, and all package __pycache__"
else
    echo "[Skip] Extended cleanup not performed."
fi

echo
echo "=== Cleanup completed ==="
