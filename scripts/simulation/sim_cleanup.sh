#!/bin/bash
# Cleanup script for SPFERE simulation
# Location: scripts/simulation/sim_cleanup.sh

set -euo pipefail

# -----------------------------------------------------
# 0. Resolve project root (two levels up)
# scripts/simulation -> ../.. -> project_root
# -----------------------------------------------------
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

echo "=== SPFERE Simulation Cleanup ==="
echo "Project root: $ROOT_DIR"
echo

# -----------------------------------------------------
# 1. First confirmation: delete basic simulation outputs
# -----------------------------------------------------
if [[ $# -ge 1 ]]; then
    confirmlog="$1"
else
    read -p "Delete all simulation outputs (simlog_*)? (y/n): " confirmlog
fi

if [[ "$confirmlog" =~ ^([yY]|[yY][eE][sS])$ ]]; then
    rm -rf simlog_* 2>/dev/null || true
    echo "[OK] Deleted simlog_*"
else
    echo "[Skip] Basic simulation outputs not deleted."
fi

echo

# -----------------------------------------------------
# 2. Extended confirmation: delete venv + data + __pycache__
# -----------------------------------------------------
if [[ $# -ge 2 ]]; then
    confirmex="$2"
else
    read -p "Delete venv (sim_venv), data/, and __pycache__? (y/n): " confirmex
fi

if [[ "$confirmex" =~ ^([yY]|[yY][eE][sS])$ ]]; then
    rm -rf sim_venv
    rm -rf data
    rm -rf src/chen2026spfere/**/__pycache__ 2>/dev/null || true
    echo "[OK] Deleted sim_venv, data/, and all __pycache__ under src/chen2026spfere"
else
    echo "[Skip] Extended cleanup not performed."
fi

echo
echo "=== Cleanup completed ==="
