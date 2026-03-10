"""Shared BBF runtime helpers for benchmark env wrapping."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


class BBFResetSemanticsEnvAdapter:
    """
    Apply BBF-native reset semantics on top of raw ALE env resets.

    Supported semantics:
    - random no-op reset startup
    - optional FIRE startup sequence when supported by the current game
    """

    _NOOP_ALE_ACTION = 0
    _FIRE_ALE_ACTION = 1
    _SECONDARY_FIRE_ALE_ACTION = 2

    def __init__(
        self,
        env,
        *,
        seed: int,
        noop_max: int = 0,
        enable_fire_reset: bool = True,
    ) -> None:
        self._env = env
        self._rng = np.random.default_rng(int(seed))
        self._noop_max = max(0, int(noop_max))
        self._enable_fire_reset = bool(enable_fire_reset)
        self.action_set = []
        self._local_by_ale: Dict[int, int] = {}
        self._noop_local_idx = 0
        self._fire_local_idx: Optional[int] = None
        self._fire_secondary_local_idx: Optional[int] = None
        self.fire_reset_supported = False
        self._refresh_action_bindings()

    def _refresh_action_bindings(self) -> None:
        self.action_set = [int(a) for a in getattr(self._env, "action_set", ())]
        self._local_by_ale = {int(a): idx for idx, a in enumerate(self.action_set)}
        self._noop_local_idx = int(self._local_by_ale.get(self._NOOP_ALE_ACTION, 0))
        self._fire_local_idx, self._fire_secondary_local_idx = self._resolve_fire_local_actions()
        self.fire_reset_supported = bool(self._fire_local_idx is not None)

    def _resolve_fire_local_actions(self) -> tuple[Optional[int], Optional[int]]:
        fire_locals = []
        meanings_fn = getattr(self._env, "get_action_meanings", None)
        if callable(meanings_fn):
            try:
                meanings = list(meanings_fn())
            except Exception:
                meanings = []
            if len(meanings) == len(self.action_set):
                for idx, meaning in enumerate(meanings):
                    if "FIRE" in str(meaning).upper():
                        fire_locals.append(int(idx))

        if not fire_locals and self._FIRE_ALE_ACTION in self._local_by_ale:
            fire_locals.append(int(self._local_by_ale[self._FIRE_ALE_ACTION]))

        if not fire_locals:
            return None, None

        primary = int(fire_locals[0])
        if len(fire_locals) >= 2:
            secondary = int(fire_locals[1])
        elif self._SECONDARY_FIRE_ALE_ACTION in self._local_by_ale:
            secondary = int(self._local_by_ale[self._SECONDARY_FIRE_ALE_ACTION])
        else:
            secondary = int(primary)
        return int(primary), int(secondary)

    def load_game(self, game_id: str):
        out = self._env.load_game(str(game_id))
        self._refresh_action_bindings()
        return out

    def lives(self) -> int:
        return int(self._env.lives())

    def step(self, action_idx: int):
        return self._env.step(int(action_idx))

    def _apply_noop_reset(self, obs_rgb):
        if self._noop_max <= 0:
            return obs_rgb
        noops = int(self._rng.integers(1, self._noop_max + 1))
        obs = obs_rgb
        for _ in range(noops):
            step = self._env.step(int(self._noop_local_idx))
            obs = step.obs_rgb
            if bool(step.terminated) or bool(step.truncated):
                obs = self._env.reset()
        return obs

    def _apply_fire_reset(self, obs_rgb):
        if not self._enable_fire_reset or self._fire_local_idx is None:
            return obs_rgb
        obs = obs_rgb
        for local_idx in (int(self._fire_local_idx), int(self._fire_secondary_local_idx or self._fire_local_idx)):
            step = self._env.step(int(local_idx))
            obs = step.obs_rgb
            if bool(step.terminated) or bool(step.truncated):
                obs = self._env.reset()
        return obs

    def reset(self):
        obs = self._env.reset()
        obs = self._apply_noop_reset(obs)
        obs = self._apply_fire_reset(obs)
        return obs

    def __getattr__(self, name: str):
        return getattr(self._env, name)
