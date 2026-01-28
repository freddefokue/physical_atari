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

"""
Centralized constants for the Physical Atari continual learning benchmark.
Single source of truth for magic numbers used across all agents.
"""

# =============================================================================
# Atari Environment Constants
# =============================================================================

# Standard frame skip (action repeat) for Atari environments
FRAME_SKIP = 4

# Full canonical action set size (all 18 Atari actions)
ATARI_CANONICAL_ACTIONS = 18

# Standard observation dimensions after preprocessing
OBSERVATION_HEIGHT = 84
OBSERVATION_WIDTH = 84

# Number of frames to stack for temporal information
FRAME_STACK = 4

# Maximum random no-op actions at episode start
NOOP_MAX = 30

# =============================================================================
# Continual Learning Benchmark Constants
# =============================================================================

# Default game list for continual benchmark (8 games)
# Default game list for continual benchmark (8 games, Carmack talk set)
DEFAULT_CONTINUAL_GAMES = (
    "MsPacmanNoFrameskip-v4",
    "CentipedeNoFrameskip-v4",
    "QbertNoFrameskip-v4",
    "DefenderNoFrameskip-v4",
    "KrullNoFrameskip-v4",
    "AtlantisNoFrameskip-v4",
    "UpNDownNoFrameskip-v4",
    "BattleZoneNoFrameskip-v4",
)

# Default number of cycles through all games
DEFAULT_CONTINUAL_CYCLES = 3

# Default frames per game per cycle
DEFAULT_CONTINUAL_CYCLE_FRAMES = 400_000

# =============================================================================
# Logging & Visualization Constants
# =============================================================================

# Number of data points for progress graphs
PROGRESS_POINTS = 1000

# Default window size for moving average calculations (in frames)
DEFAULT_AVERAGE_FRAMES = 100_000

# =============================================================================
# Buffer Constants
# =============================================================================

# Sentinel value for uninitialized buffer entries
INVALID_BUFFER_VALUE = -999.0

# Sentinel value for uninitialized integer buffer entries
INVALID_BUFFER_INT = -999
