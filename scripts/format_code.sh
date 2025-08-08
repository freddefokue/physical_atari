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

# chmod +x scripts/format_code.sh
set -e

# run from repo root dir
cd "$(dirname "$0")/.."

# ensure ~/.local/bin is in PATH
export PATH="$HOME/.local/bin:$PATH"

REQUIRED_TOOLS=("pre-commit" "black" "isort" "ruff")

# check for missing tools
missing=()
for tool in "${REQUIRED_TOOLS[@]}"; do
    if ! command -v "$tool" &>/dev/null; then
        missing+=("$tool")
    fi
done

if [ ${#missing[@]} -gt 0 ]; then
    echo "Missing tools: ${missing[*]}"
    echo "Installing dev requirements from requirements-dev.txt..."
    python3 -m pip install --user -r requirements-dev.txt
fi

echo "Running pre-commit on all files..."
pre-commit run --all-files
