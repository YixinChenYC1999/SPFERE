#!/bin/bash
# Modern client setup script for chen2026spfere client

set -euo pipefail

# ----------------------------------------
# 0. Resolve project root (three levels up)
# scripts/on_device/client -> ../../.. -> project_root
# ----------------------------------------
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

echo "=== SPFERE Client Setup ==="
echo "Project root resolved to: $ROOT_DIR"
echo

# ----------------------------------------
# 1. System packages
# ----------------------------------------
echo "[1/4] Updating system and installing required system packages..."
sudo apt-get update
sudo apt-get install -y screen rsync python3 python3-venv python3-pip git

# ----------------------------------------
# 2. Create and activate virtual environment
# ----------------------------------------
echo "[2/4] Creating Python virtual environment..."
python3 -m venv client_venv
# shellcheck disable=SC1091
source client_venv/bin/activate

echo "Python version in venv:"
python --version
echo

# ----------------------------------------
# 3. Install project (and all dependencies) via pyproject.toml
# ----------------------------------------
echo "[3/4] Upgrading pip/setuptools/wheel..."
python -m pip install --upgrade pip setuptools wheel

echo "[3/4] Installing chen2026spfere in editable mode (pip install -e .)..."
pip install -e .

# If not using raspi 4/5, comment this line to avoid errors
# python -m pip install smbus

echo

# ----------------------------------------
# 4. Create runtime directories & make scripts executable
# ----------------------------------------
echo "[4/4] Creating runtime directories..."
mkdir -p data client_log client_model

echo "Making scripts executable..."
chmod +x scripts/on_device/client/*.sh 2>/dev/null || true

# ----------------------------------------
# 5. Collect client information and write client_info.py
# ----------------------------------------
echo
echo "=== Collecting client info ==="

DEFAULT_ID="$(hostname)"
read -p "Client ID [default: $DEFAULT_ID]: " INPUT_ID
CLIENT_ID=${INPUT_ID:-$DEFAULT_ID}

echo
echo "Select device type:"
echo "  1) Ras_Pi_4"
echo "  2) Ras_Pi_5"
echo "  3) Opi5_Plus"
echo "  4) RockPi4C+"
read -p "Enter device type number: " DEV_NUM

case "$DEV_NUM" in
    1) DEVICE_TYPE="Ras_Pi_4" ;;
    2) DEVICE_TYPE="Ras_Pi_5" ;;
    3) DEVICE_TYPE="Opi5_Plus" ;;
    4) DEVICE_TYPE="RockPi4C+" ;;
    *) echo "Invalid choice. Defaulting to Ras_Pi_4"; DEVICE_TYPE="Ras_Pi_4" ;;
esac

echo
read -p "Is this device plugged in (not battery powered)? (y/n): " POWER_ANS

if [[ "$POWER_ANS" =~ ^([yY]|[yY][eE][sS])$ ]]; then
    POWER_ON=1
else
    POWER_ON=0
fi

# Write into chen2026spfere/client/client_info.py
CLIENT_INFO_FILE="src/chen2026spfere/client/client_info.py"

echo
echo "[INFO] Writing client info into: $CLIENT_INFO_FILE"

cat > "$CLIENT_INFO_FILE" <<EOF
CLIENT_INFO = {
    "client_id": "${CLIENT_ID}", # Client name
    "device_type": "${DEVICE_TYPE}", # Device type
    "power-on": ${POWER_ON}, # Power type 1 for plugin, 0 for bat
    "train-round": -1,
    "train-stage": -1,
    "train-flag": 1,
    "agg-count": 1
}
EOF

echo "[INFO] Client info saved."

echo
echo "=== Client setup complete! ==="
echo "To start the client:"
echo "  source client_venv/bin/activate"
echo "  bash scripts/on_device/client/client_test.sh"
echo
