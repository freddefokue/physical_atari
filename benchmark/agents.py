"""Agents for the single-game streaming benchmark runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Protocol

import numpy as np


class StreamingAgent(Protocol):
    """Agent interface for frame-by-frame streaming control."""

    def step(self, obs_rgb, reward, terminated, truncated, info) -> int:
        """Return an action index for the current frame."""


@dataclass
class RandomAgent:
    """Uniform random action agent over action indices."""

    num_actions: int
    seed: int = 0

    def __post_init__(self) -> None:
        if self.num_actions <= 0:
            raise ValueError("num_actions must be > 0")
        self._rng = np.random.default_rng(self.seed)

    def step(self, obs_rgb, reward, terminated, truncated, info) -> int:
        del obs_rgb, reward, terminated, truncated, info
        return int(self._rng.integers(0, self.num_actions))


@dataclass
class RepeatActionAgent:
    """Always returns the same action index."""

    action_idx: int = 0

    def step(self, obs_rgb, reward, terminated, truncated, info) -> int:
        del obs_rgb, reward, terminated, truncated, info
        return int(self.action_idx)


class FakeSequenceAgent:
    """
    Deterministic test agent.

    If `values` is provided, actions are emitted from that sequence.
    Otherwise actions increase from `start` by +1 on each call.
    """

    def __init__(
        self,
        values: Optional[Iterable[int]] = None,
        start: int = 1,
        repeat_last: bool = True,
    ) -> None:
        self._values = list(values) if values is not None else None
        self._repeat_last = repeat_last
        self._cursor = 0
        self._next_value = start

    def step(self, obs_rgb, reward, terminated, truncated, info) -> int:
        del obs_rgb, reward, terminated, truncated, info
        if self._values is None:
            value = self._next_value
            self._next_value += 1
            return int(value)

        if not self._values:
            raise ValueError("values must not be empty when provided")

        if self._cursor < len(self._values):
            value = self._values[self._cursor]
            self._cursor += 1
            return int(value)

        if self._repeat_last:
            return int(self._values[-1])
        raise IndexError("FakeSequenceAgent exhausted values and repeat_last=False")
