from __future__ import annotations

import numpy as np

from benchmark.bbf_runtime import BBFResetSemanticsEnvAdapter
from benchmark.runner import EnvStep


def test_bbf_reset_adapter_refreshes_fire_support_across_load_game_boundaries():
    class _Env:
        def __init__(self) -> None:
            self._game = "a"
            self.action_set = [0, 1, 2]
            self.step_calls = []

        def load_game(self, game_id: str):
            self._game = str(game_id)
            if self._game == "a":
                self.action_set = [0, 1, 2]
            else:
                self.action_set = [0, 3]
            return list(self.action_set)

        def get_action_meanings(self):
            if self._game == "a":
                return ["NOOP", "FIRE", "UP"]
            return ["NOOP", "RIGHT"]

        def reset(self):
            return np.zeros((4, 4, 3), dtype=np.uint8)

        def lives(self):
            return 3

        def step(self, action_idx: int):
            self.step_calls.append(int(action_idx))
            return EnvStep(
                obs_rgb=np.zeros((4, 4, 3), dtype=np.uint8),
                reward=0.0,
                terminated=False,
                truncated=False,
                lives=3,
                termination_reason=None,
            )

    env = _Env()
    wrapped = BBFResetSemanticsEnvAdapter(env, seed=0, noop_max=1, enable_fire_reset=True)

    wrapped.load_game("a")
    wrapped.reset()
    assert wrapped.fire_reset_supported is True
    assert env.step_calls == [0, 1, 2]

    env.step_calls.clear()
    wrapped.load_game("b")
    wrapped.reset()
    assert wrapped.fire_reset_supported is False
    assert env.step_calls == [0]
