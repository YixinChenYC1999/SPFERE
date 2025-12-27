#!/bin/bash
# Modern server setup script for chen2026spfere

set -euo pipefail

# -----------------------------------------------------
# 0. Resolve project root (three levels up)
# scripts/on_device/server -> ../../.. -> project_root
# -----------------------------------------------------
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

echo "=== SPFERE Server Setup ==="
echo "Project root resolved to: $ROOT_DIR"
echo

# -----------------------------------------------------
# 1. System packages
# -----------------------------------------------------
echo "[1/4] Updating system and installing required system packages..."
sudo apt-get update
sudo apt-get install -y screen rsync python3 python3-venv python3-pip git

# -----------------------------------------------------
# 2. Create and activate virtual environment
# -----------------------------------------------------
echo "[2/4] Creating Python virtual environment (server_venv)..."
python3 -m venv server_venv
# shellcheck disable=SC1091
source server_venv/bin/activate

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
mkdir -p data server_log server_model

echo "Making server-related scripts executable..."

# server-side on-device scripts
chmod +x scripts/on_device/server/*.sh 2>/dev/null || true
# optional: batch client control scripts, if present
chmod +x scripts/on_device/client_batch/*.sh 2>/dev/null || true

echo
echo "=== Server setup complete! ==="
echo "To start the server, for example:"
echo "  cd \"$ROOT_DIR\""
echo "  source server_venv/bin/activate"
echo "  bash scripts/on_device/server/server_test.sh"
echo
