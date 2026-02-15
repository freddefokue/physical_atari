"""Thin ALEInterface wrapper for single-game and multi-game Atari runs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

import numpy as np

from benchmark.runner import EnvStep

try:
    from ale_py import ALEInterface, roms
except ImportError:  # pragma: no cover - exercised only when ale_py is missing
    ALEInterface = None
    roms = None


@dataclass
class ALEEnvConfig:
    """
    ALE environment configuration.

    - `game`: default ROM key passed to `ale_py.roms.get_rom_path`.
    - `seed`: ALE random seed.
    - `sticky_action_prob`: ALE sticky action probability.
    - `full_action_space`: use full legal actions when True, minimal set when False.
    - `life_loss_termination`: mark life loss as terminated (optional, default False).
    """

    game: str
    seed: int = 0
    sticky_action_prob: float = 0.25
    full_action_space: bool = True
    life_loss_termination: bool = False


class ALEAtariEnv:
    """Direct ALEInterface env wrapper exposing raw RGB frames."""

    def __init__(self, config: ALEEnvConfig) -> None:
        if ALEInterface is None or roms is None:
            raise ImportError("ale_py is required. Install `ale-py` to use ALEAtariEnv.")

        self.config = config
        self.ale = ALEInterface()
        self.ale.setInt("random_seed", int(config.seed))
        self.ale.setFloat("repeat_action_probability", float(config.sticky_action_prob))
        self.current_game = ""
        self.action_set: List[int] = []
        self.load_game(config.game)
        self._prev_lives = int(self.ale.lives())

    @staticmethod
    def _normalize_rom_name(name: str) -> str:
        return re.sub(r"[^a-z0-9]", "", name.lower())

    @classmethod
    def _resolve_rom_path(cls, game: str):
        """
        Resolve ROM path from ale_py ROM registry.

        ale_py>=0.10 exposes `roms.get_rom_path(game)`.
        Older ale_py versions expose ROMs as module attributes (e.g. `roms.Pong`).
        """
        if hasattr(roms, "get_rom_path"):
            return roms.get_rom_path(game)

        target = cls._normalize_rom_name(game)
        candidates = []
        for attr in dir(roms):
            if attr.startswith("_"):
                continue
            value = getattr(roms, attr)
            if cls._normalize_rom_name(attr) == target:
                return value
            candidates.append(attr)
        raise ValueError(
            f"ROM '{game}' not found in ale_py ROM registry. Available names include: {', '.join(sorted(candidates)[:20])}..."
        )

    def lives(self) -> int:
        return int(self.ale.lives())

    def load_game(self, game: str) -> List[int]:
        """
        Load a ROM and refresh the active action set.

        Returns the local action set for the loaded game.
        """
        rom_path = self._resolve_rom_path(game)
        self.ale.loadROM(str(rom_path))
        self.ale.setFloat("repeat_action_probability", float(self.config.sticky_action_prob))

        if self.config.full_action_space:
            raw_action_set = self.ale.getLegalActionSet()
        else:
            raw_action_set = self.ale.getMinimalActionSet()
        self.action_set = [int(a) for a in raw_action_set]
        if not self.action_set:
            raise RuntimeError(f"No actions available for game '{game}'")

        self.current_game = str(game)
        self._prev_lives = int(self.ale.lives())
        return list(self.action_set)

    def reset(self) -> np.ndarray:
        self.ale.reset_game()
        self._prev_lives = int(self.ale.lives())
        return self.get_screen_rgb()

    def get_screen_rgb(self) -> np.ndarray:
        obs = self.ale.getScreenRGB()
        return np.asarray(obs, dtype=np.uint8)

    def step(self, action_idx: int) -> EnvStep:
        if action_idx < 0 or action_idx >= len(self.action_set):
            raise ValueError(f"action_idx {action_idx} out of bounds for action set size {len(self.action_set)}")

        ale_action = int(self.action_set[action_idx])
        reward = float(self.ale.act(ale_action))
        lives = int(self.ale.lives())
        game_over = bool(self.ale.game_over())
        life_lost = lives < self._prev_lives

        terminated = game_over or (self.config.life_loss_termination and life_lost)
        termination_reason = None
        if game_over:
            termination_reason = "game_over"
        elif terminated:
            termination_reason = "life_loss"

        self._prev_lives = lives
        return EnvStep(
            obs_rgb=self.get_screen_rgb(),
            reward=reward,
            terminated=terminated,
            truncated=False,
            lives=lives,
            termination_reason=termination_reason,
        )
