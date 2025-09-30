from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from gymnasium.spaces import Space


class Agent(ABC):
    """Minimal agent interface for the benchmark."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def begin_task(self, game_id: str, cycle: int, action_space: Space) -> None:  # noqa: D401
        """Called at the start of each game cycle."""
        return None

    def observe(
        self,
        obs: Any,
        action: int,
        reward: float,
        next_obs: Any,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> None:
        """Called after each environment step to provide transition data."""
        return None

    def end_episode(self, episode_return: float) -> None:  # noqa: D401
        """Called at the end of each episode."""
        return None

    @abstractmethod
    def act(self, obs: Any, action_space: Space) -> int:
        raise NotImplementedError
