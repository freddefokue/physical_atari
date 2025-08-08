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

# chmod +x nvidia-powerd.sh
# sudo ./nvidia-powerd.sh

# NVIDIA provides a service called 'nvidia-powerd' that enables Dynamic Boost - a feature
# that reallocates power between the CPU and GPU based on workload for better performance.
# See https://download.nvidia.com/XFree86/Linux-x86_64/510.47.03/README/dynamicboost.html.

# This must be run on host after every driver update.
# Verify power limits are no longer limited with 'nvidia-smi --query-gpu=power.draw,power.limit --format=csv'

set -e

echo "Starting NVIDIA Dynamic Boost setup..."

# get the major driver version number (e.g., "535" from "535.54.03")
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d '.' -f1)

if [[ -z "$DRIVER_VER" ]]; then
  echo "[ERROR] Could not determine NVIDIA driver version. Verify driver installed with nvidia-smi."
  exit 1
fi

echo "Detected NVIDIA driver version: $DRIVER_VER"

# paths to the source files in the driver docs
DBUS_CONF_SRC="/usr/share/doc/nvidia-driver-$DRIVER_VER/nvidia-dbus.conf"
POWERD_SERVICE_SRC="/usr/share/doc/nvidia-kernel-common-$DRIVER_VER/nvidia-powerd.service"

# Verify source files exist
if [[ ! -f "$DBUS_CONF_SRC" ]]; then
  echo "[ERROR] $DBUS_CONF_SRC not found."
  exit 1
fi

if [[ ! -f "$POWERD_SERVICE_SRC" ]]; then
  echo "[ERROR] $POWERD_SERVICE_SRC not found."
  exit 1
fi

DBUS_CONF_DST="/etc/dbus-1/system.d/nvidia-dbus.conf"
POWERD_SERVICE_DST="/etc/systemd/system/nvidia-powerd.service"

echo "Copying $DBUS_CONF_SRC to $DBUS_CONF_DST"
cp "$DBUS_CONF_SRC" "$DBUS_CONF_DST"

echo "Copying $POWERD_SERVICE_SRC to $POWERD_SERVICE_DST"
cp "$POWERD_SERVICE_SRC" "$POWERD_SERVICE_DST"

echo "Reloading systemd daemon..."
systemctl daemon-reload

echo "Enabling and starting nvidia-powerd.service..."
systemctl enable --now nvidia-powerd.service

echo "NVIDIA powerd service status:"
systemctl status nvidia-powerd.service --no-pager
