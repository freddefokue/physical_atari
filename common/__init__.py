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
Common utilities for Physical Atari continual learning agents.

This module provides shared functionality:
- constants: Centralized magic numbers and configuration values
- env_utils: Environment creation utilities
- metrics: Moving average, progress graphs, and summary writing
- logging_utils: Consistent logging across agents
"""

from .constants import (
    # Atari environment constants
    FRAME_SKIP,
    ATARI_CANONICAL_ACTIONS,
    OBSERVATION_HEIGHT,
    OBSERVATION_WIDTH,
    FRAME_STACK,
    NOOP_MAX,
    # Continual learning benchmark constants
    DEFAULT_CONTINUAL_GAMES,
    DEFAULT_CONTINUAL_CYCLES,
    DEFAULT_CONTINUAL_CYCLE_FRAMES,
    # Logging & visualization constants
    PROGRESS_POINTS,
    DEFAULT_AVERAGE_FRAMES,
    # Buffer constants
    INVALID_BUFFER_VALUE,
    INVALID_BUFFER_INT,
)

from .env_utils import make_atari_env

from .metrics import (
    moving_average,
    update_progress_graphs,
    write_continual_summary,
)

from .logging_utils import (
    AgentLogger,
    create_logger,
    log_to_file,
)

from .profiling import (
    nvtx_range,
    nvtx_mark,
    ProfilerSection,
    ProfileSections,
)


__all__ = [
    # Constants
    "FRAME_SKIP",
    "ATARI_CANONICAL_ACTIONS",
    "OBSERVATION_HEIGHT",
    "OBSERVATION_WIDTH",
    "FRAME_STACK",
    "NOOP_MAX",
    "DEFAULT_CONTINUAL_GAMES",
    "DEFAULT_CONTINUAL_CYCLES",
    "DEFAULT_CONTINUAL_CYCLE_FRAMES",
    "PROGRESS_POINTS",
    "DEFAULT_AVERAGE_FRAMES",
    "INVALID_BUFFER_VALUE",
    "INVALID_BUFFER_INT",
    # Environment utilities
    "make_atari_env",
    # Metrics
    "moving_average",
    "update_progress_graphs",
    "write_continual_summary",
    # Logging
    "AgentLogger",
    "create_logger",
    "log_to_file",
    # Profiling
    "nvtx_range",
    "nvtx_mark",
    "ProfilerSection",
    "ProfileSections",
]
