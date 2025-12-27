#!/bin/bash
# Simulation setup script for chen2026spfere

set -euo pipefail

# -----------------------------------------------------
# 0. Resolve project root (two levels up)
# scripts/simulation -> ../.. -> project_root
# -----------------------------------------------------
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

echo "=== SPFERE Simulation Setup ==="
echo "Project root resolved to: $ROOT_DIR"
echo

# -----------------------------------------------------
# 1. System packages
# -----------------------------------------------------
echo "[1/4] Updating system and installing required system packages..."
sudo apt-get update
sudo apt-get install -y screen rsync parallel python3 python3-venv python3-pip git

# -----------------------------------------------------
# 2. Create and activate virtual environment
# -----------------------------------------------------
echo "[2/4] Creating Python virtual environment (sim_venv)..."
python3 -m venv sim_venv
# shellcheck disable=SC1091
source sim_venv/bin/activate

echo "Python version in venv:"
python --version
echo

# -----------------------------------------------------
# 3. Install project (and all dependencies) via pyproject.toml
# -----------------------------------------------------
echo "[3/4] Upgrading pip/setuptools/wheel..."
python -m pip install --upgrade pip setuptools wheel

echo "[3/4] Installing chen2026spfere in editable mode (pip install -e .)..."
pip install -e .
echo

# -----------------------------------------------------
# 4. Create runtime directories & make scripts executable
# -----------------------------------------------------
echo "[4/4] Creating runtime directories..."
mkdir -p data

echo "Making simulation scripts executable..."
chmod +x scripts/simulation/*.sh 2>/dev/null || true

echo
echo "=== Simulation setup is complete. ==="
echo "To run a simulation, for example:"
echo "  cd \"$ROOT_DIR\""
echo "  source sim_venv/bin/activate"
echo "  bash scripts/simulation/sim_test.sh"
echo
