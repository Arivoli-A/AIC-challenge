#!/bin/bash
set -e

# ===== CONFIG =====
NEW_DOCKER_DIR="/mnt/disks/docker"

echo "🚀 Stopping Docker..."
sudo systemctl stop docker

echo "📁 Creating new Docker directory at $NEW_DOCKER_DIR ..."
sudo mkdir -p "$NEW_DOCKER_DIR"
sudo chmod -R 755 /mnt/disks/docker

echo "📦 Copying existing Docker data (this may take time)..."
sudo rsync -aHAX /var/lib/docker/ "$NEW_DOCKER_DIR/"

echo "⚙️ Writing Docker config (/etc/docker/daemon.json)..."
sudo mkdir -p /etc/docker

sudo tee /etc/docker/daemon.json > /dev/null <<EOF
{
  "data-root": "$NEW_DOCKER_DIR"
}
EOF

echo "🔄 Reloading systemd..."
sudo systemctl daemon-reload

echo "▶️ Starting Docker..."
sudo systemctl start docker

echo "🔍 Verifying Docker root directory..."
docker info | grep "Docker Root Dir" || true

echo "✅ Done! Docker is now using: $NEW_DOCKER_DIR"