#!/bin/bash
TARGET_IFACE="wlan0"
SERVICE_NAME="disable-${TARGET_IFACE}-powersave.service"
SYSTEMD_PATH="/etc/systemd/system/${SERVICE_NAME}"

IW_BIN="$(command -v iw || echo /usr/sbin/iw)"
IWCONF_BIN="$(command -v iwconfig || true)"

if [[ -x "$IW_BIN" ]]; then
  OFF_CMD="$IW_BIN dev ${TARGET_IFACE} set power_save off"
elif [[ -n "$IWCONF_BIN" ]]; then
  OFF_CMD="$IWCONF_BIN ${TARGET_IFACE} power off"
else
  echo "[ERROR] Neither 'iw' nor 'iwconfig' found."
  exit 1
fi

sudo tee "$SYSTEMD_PATH" > /dev/null <<EOF
[Unit]
Description=Disable power saving for ${TARGET_IFACE} (wait-ready)
Wants=network-online.target NetworkManager.service
After=network-online.target NetworkManager.service NetworkManager-wait-online.service

[Service]
Type=oneshot

ExecStartPre=/usr/bin/env bash -c 'for i in {1..30}; do ip link show ${TARGET_IFACE} >/dev/null 2>&1 && exit 0; sleep 1; done; exit 1'

ExecStartPre=/usr/bin/env bash -c 'for i in {1..15}; do s=$(cat /sys/class/net/${TARGET_IFACE}/operstate 2>/dev/null || echo down); [[ "$s" = up ]] && exit 0; sleep 1; done; exit 0'

ExecStart=/usr/bin/sleep 10

ExecStart=${OFF_CMD}
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now "$SERVICE_NAME"
