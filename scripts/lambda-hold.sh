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

# To run:
# lambda-hold.sh — prevent updates to Lambda Stack packages
# To unhold by package: sudo apt-mark unhold <package_name>
# or:
# to unhold all: apt-mark showhold | xargs sudo apt-mark unhold

set -e

echo "Locking Lambda Stack packages..."

# core packages to hold
LAMBDA_PACKAGES=(
  lambda-stack-cuda
  lambda-stack
  libcudnn8
  libcudnn8-dev
  libnccl2
  libnccl-dev
  nvidia-cuda-toolkit
  nvidia-driver
  nvidia-headless
  nvidia-fabricmanager
  cuda
)

# go through packages, and if installed, hold
for pkg in "${LAMBDA_PACKAGES[@]}"; do
  MATCHED_PKGS=$(dpkg -l | awk '{print $2}' | grep -E "^${pkg}.*$" || true)
  for matched in $MATCHED_PKGS; do
    echo "-- Holding $matched"
    sudo apt-mark hold "$matched"
  done
done

echo ""
echo "Hold complete. The following packages are currently on hold:"
apt-mark showhold
