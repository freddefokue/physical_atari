#!/bin/bash
# Copyright 2025 Keen Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# chmod +x nvidia-persistence.sh
# sudo ./nvidia-persistence.sh

set -e

echo "Starting NVIDIA persistence mode setup..."

# check if nvidia-smi exists
if ! command -v nvidia-smi &> /dev/null; then
  echo "[ERROR] nvidia-smi not found. Please install NVIDIA drivers first."
  exit 1
fi

# verify script is run as root
if [[ $EUID -ne 0 ]]; then
  echo "[ERROR] Script must be run as root (e.g., sudo $0)"
  exit 1
fi

echo "Enabling persistence mode now..."
nvidia-smi -pm 1

# create systemd service to enable persistence mode on boot
SERVICE_PATH="/etc/systemd/system/nvidia-persistence.service"

echo "Creating systemd service at $SERVICE_PATH..."

cat <<EOF > "$SERVICE_PATH"
[Unit]
Description=NVIDIA Persistence Mode
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/usr/bin/nvidia-smi -pm 1
RemainAfterExit=true

[Install]
WantedBy=multi-user.target
EOF

# reload systemd and enable the service
echo "Reloading systemd daemon and enabling nvidia-persistence.service..."
systemctl daemon-reload
systemctl enable nvidia-persistence.service
systemctl start nvidia-persistence.service

# verify status
echo "Verifying persistence mode status..."
nvidia-smi -q | grep "Persistence Mode"

echo "NVIDIA persistence mode enabled and service installed."
