#!/bin/bash
# Cleanup script for SPFERE client

set -euo pipefail

# -----------------------------------------------------
# 0. Resolve project root (three levels up)
# -----------------------------------------------------
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

echo "=== SPFERE Client Cleanup ==="
echo "Project root: $ROOT_DIR"
echo

# -----------------------------------------------------
# 1. First confirmation: delete basic outputs?
# -----------------------------------------------------
if [[ $# -ge 1 ]]; then
    confirmlog="$1"
else
    read -p "Delete all basic output (client_log*, client_model*)? (y/n): " confirmlog
fi

if [[ "$confirmlog" =~ ^([yY]|[yY][eE][sS])$ ]]; then
    rm -rf client_model*
    rm -rf client_log*
    echo "[OK] Deleted client_model* and client_log*"
else
    echo "[Skip] Basic output not deleted."
fi

echo

# -----------------------------------------------------
# 2. Second confirmation: delete venv, data, caches?
# -----------------------------------------------------
if [[ $# -ge 2 ]]; then
    confirmex="$2"
else
    read -p "Delete venv, data/, and __pycache__? (y/n): " confirmex
fi

if [[ "$confirmex" =~ ^([yY]|[yY][eE][sS])$ ]]; then
    rm -rf client_venv
    rm -rf data
    rm -rf src/chen2026spfere/**/__pycache__ 2>/dev/null || true
    echo "[OK] Deleted venv, data/, and all __pycache__ in package tree"
else
    echo "[Skip] Extended cleanup not performed."
fi

echo
echo "=== Cleanup completed ==="
