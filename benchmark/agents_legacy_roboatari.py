"""Thin benchmark adapters for legacy RoboAtari ``Agent.frame(...)`` agents."""

from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Tuple


def _to_json_scalar(value: Any) -> Any:
    if isinstance(value, (int, float, bool, str)) or value is None:
        return value
    item_fn = getattr(value, "item", None)
    if callable(item_fn):
        try:
            item = item_fn()
        except Exception:  # pragma: no cover - defensive
            return str(value)
        if isinstance(item, (int, float, bool, str)) or item is None:
            return item
    return str(value)


@contextmanager
def _roboatari_import_path() -> Iterator[None]:
    repo_root = Path(__file__).resolve().parents[2]
    roboatari_root = repo_root / "roboatari"
    inserted = False
    root_str = str(roboatari_root)
    if roboatari_root.exists() and root_str not in sys.path:
        sys.path.insert(0, root_str)
        inserted = True
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(root_str)
            except ValueError:  # pragma: no cover - defensive
                pass


class LegacyRoboAtariAdapter:
    """Benchmark shim around legacy RoboAtari frame-based agents."""

    def __init__(
        self,
        *,
        agent_name: str,
        module_name: str,
        import_error_hint: str,
        data_dir: str,
        seed: int,
        num_actions: int,
        total_frames: int,
        agent_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if num_actions <= 0:
            raise ValueError("num_actions must be > 0")

        self._agent_name = str(agent_name)
        self._module_name = str(module_name)
        self._import_error_hint = str(import_error_hint)
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
            with _roboatari_import_path():
                module = importlib.import_module(self._module_name)
        except Exception as exc:  # pragma: no cover - optional dependency stack
            raise ImportError(self._import_error_hint) from exc

        agent_cls = getattr(module, "Agent", None)
        if agent_cls is None:
            raise ImportError(f"{self._module_name} does not expose Agent")

        self._agent = agent_cls(
            data_dir=str(data_dir),
            seed=int(seed),
            num_actions=int(num_actions),
            total_frames=int(total_frames),
            **self._agent_kwargs,
        )

    @staticmethod
    def _map_legacy_end_of_episode_code(boundary: Mapping[str, Any], *, terminated: bool, truncated: bool) -> int:
        for key in ("boundary_cause", "termination_reason", "env_termination_reason"):
            raw_value = boundary.get(key)
            if raw_value is None:
                continue
            normalized = str(raw_value).strip().lower()
            if normalized in LegacyRoboAtariAdapter._LEGACY_END_OF_EPISODE_CODES:
                return int(LegacyRoboAtariAdapter._LEGACY_END_OF_EPISODE_CODES[normalized])
            if "life_loss" in normalized:
                return 1
            if "timeout" in normalized or "truncat" in normalized or "visit_switch" in normalized:
                return 3
        if terminated:
            return 2
        if truncated:
            return 3
        return 0

    @staticmethod
    def _parse_boundary(boundary: Any) -> Tuple[bool, bool, int]:
        if isinstance(boundary, Mapping):
            terminated = bool(boundary.get("terminated", False))
            truncated = bool(boundary.get("truncated", False))
            if "legacy_end_of_episode" in boundary:
                end_of_episode = int(boundary["legacy_end_of_episode"])
            elif "end_of_episode_code" in boundary:
                end_of_episode = int(boundary["end_of_episode_code"])
            else:
                end_of_episode = LegacyRoboAtariAdapter._map_legacy_end_of_episode_code(
                    boundary,
                    terminated=terminated,
                    truncated=truncated,
                )
                if end_of_episode <= 0 and "end_of_episode_pulse" in boundary:
                    end_of_episode = int(bool(boundary["end_of_episode_pulse"]))
            return terminated, truncated, end_of_episode
        ended = bool(boundary)
        return ended, False, int(ended)

    def frame(self, obs_rgb, reward, boundary) -> int:
        _terminated, _truncated, end_of_episode = self._parse_boundary(boundary)
        agent_obs = obs_rgb
        if isinstance(boundary, Mapping) and "transition_obs_rgb" in boundary:
            agent_obs = boundary["transition_obs_rgb"]
        action_idx = int(self._agent.frame(agent_obs, float(reward), int(end_of_episode)))
        self._decision_steps += 1
        if action_idx < 0 or action_idx >= self._num_actions:
            raise ValueError(
                f"{self._agent_name} produced out-of-bounds action {action_idx} for action_space={self._num_actions}"
            )
        self._last_action_idx = int(action_idx)
        return int(action_idx)

    def step(self, obs_rgb, reward, terminated, truncated, info) -> int:
        boundary: Dict[str, Any] = {
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "end_of_episode_pulse": bool(terminated) or bool(truncated),
        }
        if isinstance(info, Mapping):
            for key in (
                "legacy_end_of_episode",
                "end_of_episode_code",
                "boundary_cause",
                "termination_reason",
                "env_termination_reason",
                "transition_obs_rgb",
                "reset_obs_rgb",
            ):
                if key in info:
                    boundary[key] = info[key]
        return self.frame(obs_rgb, reward, boundary)

    def get_config(self) -> Dict[str, Any]:
        return dict(self._config)

    def get_stats(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            "decision_steps": int(self._decision_steps),
            "last_action_idx": int(self._last_action_idx),
        }
        for owner in (self._agent, getattr(self._agent, "core", None)):
            if owner is None:
                continue
            for key in (
                "frame_count",
                "training_steps",
                "epsilon",
                "last_loss",
                "loss_ema",
                "last_avg_q",
                "last_max_q",
                "last_td_error",
                "last_grad_norm",
                "step_count",
                "global_step",
                "training_step",
                "episode_reward",
                "episode_length",
            ):
                if key not in stats and hasattr(owner, key):
                    stats[str(key)] = _to_json_scalar(getattr(owner, key))
        return stats
    _LEGACY_END_OF_EPISODE_CODES = {
        "life_loss": 1,
        "life_loss_reset": 1,
        "terminated": 2,
        "game_over": 2,
        "truncated": 3,
        "timeout": 3,
        "no_reward_timeout": 3,
        "visit_switch": 3,
    }
