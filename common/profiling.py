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
NVTX profiling utilities for systematic performance analysis.

Usage with NVIDIA Nsight Systems:
    nsys profile -o profile_output python agent_dqn.py --env-id BreakoutNoFrameskip-v4
    nsys-ui profile_output.nsys-rep

These utilities provide consistent profiling markers across all agents.
"""

from __future__ import annotations

import torch
from contextlib import contextmanager
from typing import Optional


@contextmanager
def nvtx_range(name: str, color: Optional[str] = None):
    """
    Context manager for NVTX profiling ranges.
    
    Creates a labeled region in the NVIDIA Nsight Systems timeline.
    Safe to use even when CUDA is not available (becomes a no-op).
    
    Args:
        name: Label for the profiling region (e.g., "train", "act", "sample_batch")
        color: Optional color hint (not all profilers support this)
    
    Example:
        with nvtx_range("forward_pass"):
            output = model(input)
    """
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()


def nvtx_mark(name: str):
    """
    Place a single marker in the timeline (for events, not ranges).
    
    Args:
        name: Label for the marker
    """
    if torch.cuda.is_available():
        torch.cuda.nvtx.mark(name)


class ProfilerSection:
    """
    Reusable profiler section for repeated operations.
    
    More efficient than nvtx_range for hot paths as it avoids
    context manager overhead.
    
    Example:
        profiler = ProfilerSection("train_step")
        
        for batch in batches:
            profiler.start()
            # ... training code ...
            profiler.end()
    """
    
    def __init__(self, name: str):
        self.name = name
        self.enabled = torch.cuda.is_available()
    
    def start(self):
        if self.enabled:
            torch.cuda.nvtx.range_push(self.name)
    
    def end(self):
        if self.enabled:
            torch.cuda.nvtx.range_pop()


# Pre-defined section names for consistency across agents
class ProfileSections:
    """Standard profiling section names for consistent analysis."""
    
    # Action selection
    ACT = "act"
    ACT_NETWORK = "act/network_forward"
    ACT_EPSILON = "act/epsilon_check"
    
    # Training
    TRAIN = "train"
    TRAIN_SAMPLE = "train/sample_batch"
    TRAIN_FORWARD = "train/forward"
    TRAIN_LOSS = "train/compute_loss"
    TRAIN_BACKWARD = "train/backward"
    TRAIN_OPTIMIZER = "train/optimizer_step"
    TRAIN_TARGET_UPDATE = "train/target_update"
    
    # Environment interaction
    ENV_STEP = "env/step"
    ENV_RESET = "env/reset"
    
    # Buffer operations
    BUFFER_ADD = "buffer/add"
    BUFFER_SAMPLE = "buffer/sample"
    
    # BBF-specific
    SPR_LOSS = "train/spr_loss"
    AUGMENT = "train/augment"
    
    # Rainbow-specific
    PER_UPDATE = "buffer/per_update"
