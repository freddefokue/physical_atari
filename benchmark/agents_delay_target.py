"""Adapter for using root-level agent_delay_target.Agent in benchmark runners."""

from __future__ import annotations

import importlib
from typing import Any, Dict, Mapping, Optional


def _to_json_scalar(value: Any) -> Any:
    if isinstance(value, (int, float, bool, str)) or value is None:
        return value
    item_fn = getattr(value, "item", None)
    if callable(item_fn):
        try:
            item = item_fn()
            if isinstance(item, (int, float, bool, str)) or item is None:
                return item
        except Exception:  # pragma: no cover - defensive
            return str(value)
    return str(value)


class DelayTargetAdapter:
    """Benchmark StreamingAgent shim around `agent_delay_target.Agent`."""

    def __init__(
        self,
        *,
        data_dir: str,
        seed: int,
        num_actions: int,
        total_frames: int,
        agent_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if num_actions <= 0:
            raise ValueError("num_actions must be > 0")
        self._num_actions = int(num_actions)
        self._decision_steps = 0
        self._last_action_idx = 0
        self._agent_kwargs = dict(agent_kwargs or {})
        self._config = {
            "seed": int(seed),
            "num_actions": int(num_actions),
            "total_frames": int(total_frames),
            **self._agent_kwargs,
        }

        try:
            module = importlib.import_module("agent_delay_target")
        except Exception as exc:  # pragma: no cover - depends on optional dependency stack
            raise ImportError(
                "agent=delay_target requires importing root module agent_delay_target.py and its dependencies."
            ) from exc

        agent_cls = getattr(module, "Agent", None)
        if agent_cls is None:
            raise ImportError("agent_delay_target module does not expose Agent")

        self._agent = agent_cls(
            data_dir=str(data_dir),
            seed=int(seed),
            num_actions=int(num_actions),
            total_frames=int(total_frames),
            **self._agent_kwargs,
        )

    def frame(self, obs_rgb, reward, end_of_episode) -> int:
        action_idx = int(self._agent.frame(obs_rgb, float(reward), end_of_episode))
        self._decision_steps += 1
        if action_idx < 0 or action_idx >= self._num_actions:
            raise ValueError(f"delay_target produced out-of-bounds action {action_idx} for action_space={self._num_actions}")
        self._last_action_idx = int(action_idx)
        return int(action_idx)

    def step(self, obs_rgb, reward, terminated, truncated, info) -> int:
        del info
        end_of_episode = int(bool(terminated) or bool(truncated))
        return self.frame(obs_rgb, reward, end_of_episode)

    def get_config(self) -> Dict[str, Any]:
        return dict(self._config)

    def get_stats(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            "decision_steps": int(self._decision_steps),
            "last_action_idx": int(self._last_action_idx),
        }
        for key in (
            "frame_count",
            "u",
            "episode_number",
            "train_loss_ema",
            "avg_error_ema",
            "max_error_ema",
            "target_ema",
        ):
            if hasattr(self._agent, key):
                stats[str(key)] = _to_json_scalar(getattr(self._agent, key))
        return stats
