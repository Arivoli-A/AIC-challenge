#!/usr/bin/env bash

set -e

# ===== CONFIG =====

DATA_ROOT="/mnt/disks/aic-challenge/docker2"
MOUNT_POINT="/mnt/disks/aic-challenge"
SOCKET="/var/run/docker2.sock"
PIDFILE="/var/run/docker2.pid"
SERVICE_NAME="docker2"

# ===== CREATE DATA DIRECTORY =====

echo "Creating data directory at $DATA_ROOT..."
sudo mkdir -p "$DATA_ROOT"
sudo chown root:root "$DATA_ROOT"

# ===== CREATE SYSTEMD SERVICE =====

echo "Creating systemd service: $SERVICE_NAME..."

SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

sudo bash -c "cat > $SERVICE_FILE" <<EOF
[Unit]
Description=Second Docker Daemon
After=network.target
RequiresMountsFor=$MOUNT_POINT

[Service]
ExecStartPre=/bin/bash -c 'mountpoint -q $MOUNT_POINT'
ExecStart=/usr/bin/dockerd \
--data-root=$DATA_ROOT \
-H unix://$SOCKET \
--pidfile=$PIDFILE
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# ===== RELOAD SYSTEMD =====

echo "Reloading systemd..."
sudo systemctl daemon-reexec
sudo systemctl daemon-reload

# ===== ENABLE SERVICE (won't start unless disk is mounted) =====

echo "Enabling $SERVICE_NAME..."
sudo systemctl enable $SERVICE_NAME

# ===== OPTIONAL: START IF MOUNT EXISTS =====

if mountpoint -q "$MOUNT_POINT"; then
echo "Disk is mounted. Starting $SERVICE_NAME..."
sudo systemctl start $SERVICE_NAME
else
echo "Disk not mounted. Service will start automatically when available."
fi

# ===== CREATE DOCKER CONTEXT =====

echo "Creating Docker context: docker2..."

docker context create docker2 --docker "host=unix://$SOCKET" || true

# ===== DONE =====

echo ""
echo "✅ Setup complete!"
echo ""
echo "Use the new daemon with:"
echo "  docker context use docker2"
echo ""
echo "If disk is attached later, start manually:"
echo "  sudo systemctl start $SERVICE_NAME"
echo ""
