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

# chmod +x cpu-governor.sh
# sudo ./cpu-governor.sh

# Set CPU governor to performance.

set -e

GOV="performance"

echo "Starting CPU governor setup..."

# verify script is run as root
if [[ $EUID -ne 0 ]]; then
  echo "[ERROR] Script must be run as root (e.g., sudo $0)"
  exit 1
fi

# install required tools
echo "Installing cpupower tools..."
apt update
apt install -y linux-tools-common linux-tools-$(uname -r)

# disable interfering services
if systemctl is-active --quiet power-profiles-daemon.service; then
  echo "Disabling power-profiles-daemon to avoid conflicts..."
  systemctl mask power-profiles-daemon.service
  systemctl stop power-profiles-daemon.service
fi

# Set governor temporarily on all CPU cores
echo "Setting CPU governor to $GOV temporarily..."
for gov_file in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor; do
  if [ -w "$gov_file" ]; then
    echo "$GOV" > "$gov_file" || echo "[WARN] Failed to set $gov_file"
  else
    echo "[WARN] No write permission for $gov_file"
  fi
done

# create cpupower systemd service for persistence
echo "Creating cpupower systemd service..."
cat <<EOF > /etc/systemd/system/cpupower.service
[Unit]
Description=Set CPU frequency scaling governor
After=multi-user.target

[Service]
Type=oneshot
RemainAfterExit=true
ExecStart=/usr/bin/cpupower frequency-set -g $GOV

[Install]
WantedBy=multi-user.target
EOF

# reload systemd, enable and start the service
echo "Enabling and starting cpupower.service..."
systemctl daemon-reexec
systemctl daemon-reload
systemctl enable cpupower.service
systemctl start cpupower.service

# confirm status
echo "Verifying current governor:"
cpupower frequency-info | grep "governor"

echo "Success. CPU governor set to '$GOV' and service installed for persistence."
