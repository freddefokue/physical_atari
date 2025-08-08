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

# chmod +x power-profile.sh
# sudo ./power-profile.sh

# Set platform profile to performance using ACPI interface.
set -e

PROFILE_PATH="/sys/firmware/acpi/platform_profile"
PROFILE="performance"

echo "Starting power profile setup..."

# verify script is run as root
if [[ $EUID -ne 0 ]]; then
  echo "[ERROR] Script must be run as root (e.g., sudo $0)"
  exit 1
fi

# check if the platform profile interface exists
if [[ ! -w "$PROFILE_PATH" ]]; then
  echo "[ERROR] ACPI platform profile interface not available or not writable: $PROFILE_PATH"
  echo "This system may not support ACPI performance profiles."
  exit 1
fi

# disable interfering services
if systemctl is-active --quiet power-profiles-daemon.service; then
  echo "Disabling power-profiles-daemon to avoid conflicts..."
  systemctl mask power-profiles-daemon.service
  systemctl stop power-profiles-daemon.service
fi

# apply the setting immediately
echo "Setting platform profile to '$PROFILE'..."
echo "$PROFILE" > "$PROFILE_PATH"

# create a systemd service to enforce setting at boot
echo "Creating systemd service for platform profile persistence..."

cat <<EOF > /etc/systemd/system/acpi-performance-profile.service
[Unit]
Description=Set ACPI platform profile to performance
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/bin/bash -c 'echo performance > /sys/firmware/acpi/platform_profile'
RemainAfterExit=true

[Install]
WantedBy=multi-user.target
EOF

# enable and start the service
echo "Enabling and starting acpi-performance-profile.service..."
systemctl daemon-reexec
systemctl daemon-reload
systemctl enable acpi-performance-profile.service
systemctl start acpi-performance-profile.service

# confirm result
current=$(cat "$PROFILE_PATH")
echo "Current platform profile: $current"

if [[ "$current" == "$PROFILE" ]]; then
  echo "Success. Platform profile set to '$PROFILE' and will persist on reboot."
else
  echo "[WARN] Failed to apply platform profile. Current value: $current"
fi
