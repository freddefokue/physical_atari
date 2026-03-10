"""Adapter for using root-level agent_bbf.Agent in benchmark runners."""

from __future__ import annotations

import importlib
import math
import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

BBF_ACTION_REPEAT = 4
BBF_OBS_SIZE = 84


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


@dataclass
class BBFAdapterConfig:
    """Subset of BBF knobs exposed through benchmark CLIs."""

    learning_starts: int = 2_000
    buffer_size: int = 200_000
    batch_size: int = 32
    replay_ratio: int = 64
    reset_interval: int = 20_000
    no_resets_after: int = 100_000
    use_per: bool = True
    use_amp: bool = False
    torch_compile: bool = False
    torch_compile_mode: str = "reduce-overhead"

    def __post_init__(self) -> None:
        if self.learning_starts < 0:
            raise ValueError("bbf_learning_starts must be >= 0")
        if self.buffer_size <= 0:
            raise ValueError("bbf_buffer_size must be > 0")
        if self.batch_size <= 0:
            raise ValueError("bbf_batch_size must be > 0")
        if self.replay_ratio < 0:
            raise ValueError("bbf_replay_ratio must be >= 0")
        if self.reset_interval < 0:
            raise ValueError("bbf_reset_interval must be >= 0")
        if self.no_resets_after < 0:
            raise ValueError("bbf_no_resets_after must be >= 0")

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BBFAgentAdapter:
    """
    Benchmark adapter around root-level ``agent_bbf.Agent``.

    The benchmark emits raw ALE RGB frames at one frame per callback. BBF expects
    Atari-like transitions with reward clipping, 84x84 grayscale frames, 4-frame
    action repeat cadence, and 4-frame internal state stacking. This adapter
    reconstructs those semantics from the benchmark stream.
    """

    def __init__(
        self,
        *,
        seed: int,
        num_actions: int,
        total_frames: int,
        config: Optional[BBFAdapterConfig] = None,
        full_action_space: bool = True,
        parity_mode: bool = False,
        action_space_mode: str = "canonical_full",
        native_reset_semantics: bool = False,
    ) -> None:
        if num_actions <= 0:
            raise ValueError("num_actions must be > 0")
        if total_frames <= 0:
            raise ValueError("total_frames must be > 0")

        self._num_actions = int(num_actions)
        self._seed = int(seed)
        self._total_frames = int(total_frames)
        self._config = config or BBFAdapterConfig()
        self._agent_full_action_space = bool(full_action_space)
        self._parity_mode = bool(parity_mode)
        self._action_space_mode = str(action_space_mode)
        self._native_reset_semantics = bool(native_reset_semantics)

        bbf_module = self._import_bbf_module()
        gym_module = self._import_gymnasium_module()

        agent_cls = getattr(bbf_module, "Agent", None)
        cfg_cls = getattr(bbf_module, "AgentConfig", None)
        if agent_cls is None or cfg_cls is None:
            raise ImportError("agent_bbf module must expose Agent and AgentConfig")

        total_steps = int(max(1, math.ceil(float(self._total_frames) / float(BBF_ACTION_REPEAT))))
        self._bbf_config = cfg_cls(
            seed=int(self._seed),
            total_steps=total_steps,
            full_action_space=bool(self._agent_full_action_space),
            learning_starts=int(self._config.learning_starts),
            buffer_size=int(self._config.buffer_size),
            batch_size=int(self._config.batch_size),
            replay_ratio=int(self._config.replay_ratio),
            reset_interval=int(self._config.reset_interval),
            no_resets_after=int(self._config.no_resets_after),
            use_per=bool(self._config.use_per),
            use_amp=bool(self._config.use_amp),
            torch_compile=bool(self._config.torch_compile),
            torch_compile_mode=str(self._config.torch_compile_mode),
            capture_video=False,
            track=False,
            continual=False,
        )

        obs_space = gym_module.spaces.Box(low=0, high=255, shape=(BBF_OBS_SIZE, BBF_OBS_SIZE), dtype=np.uint8)
        action_space = gym_module.spaces.Discrete(int(self._num_actions))
        self._agent = agent_cls(obs_space, action_space, self._bbf_config)

        self._resize_rows: Optional[np.ndarray] = None
        self._resize_cols: Optional[np.ndarray] = None
        self._source_hw: Optional[Tuple[int, int]] = None

        self._interval_reward = 0.0
        self._interval_count = 0
        self._interval_last_raw: Optional[np.ndarray] = None
        self._interval_prev_raw: Optional[np.ndarray] = None

        self._held_action_idx = 0
        self._has_held_action = False

        self._raw_frames = 0
        self._decision_steps = 0
        self._transition_steps = 0
        self._last_train_stats: Optional[Dict[str, Any]] = None

    @staticmethod
    def _import_bbf_module():
        try:
            return importlib.import_module("agent_bbf")
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise ImportError(
                "agent=bbf requires importing root module agent_bbf.py and its dependencies (torch, gymnasium)."
            ) from exc

    @staticmethod
    def _import_gymnasium_module():
        try:
            return importlib.import_module("gymnasium")
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise ImportError("agent=bbf requires gymnasium to construct BBF spaces.") from exc

    @staticmethod
    def _clip_reward(value: float) -> float:
        if not np.isfinite(float(value)):
            return 0.0
        return float(np.sign(float(value)))

    @staticmethod
    def _parse_boundary(boundary: Any) -> Tuple[bool, bool, bool]:
        if isinstance(boundary, Mapping):
            terminated = bool(boundary.get("terminated", False))
            truncated = bool(boundary.get("truncated", False))
            episode_pulse = bool(boundary.get("end_of_episode_pulse", terminated or truncated))
            return terminated, truncated, episode_pulse
        ended = bool(boundary)
        return ended, False, ended

    def _preprocess_frame(self, obs_rgb: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs_rgb)
        if obs.dtype != np.uint8:
            obs = obs.astype(np.uint8)

        if obs.ndim == 2:
            gray = obs
        elif obs.ndim == 3 and obs.shape[-1] >= 3:
            r = obs[..., 0].astype(np.uint16)
            g = obs[..., 1].astype(np.uint16)
            b = obs[..., 2].astype(np.uint16)
            gray = ((77 * r + 150 * g + 29 * b) >> 8).astype(np.uint8)
        elif obs.ndim == 3 and obs.shape[-1] == 1:
            gray = obs[..., 0].astype(np.uint8, copy=False)
        else:
            raise ValueError(f"Unsupported observation shape for BBF adapter: {obs.shape}")

        height, width = int(gray.shape[0]), int(gray.shape[1])
        source_hw = (height, width)
        if self._source_hw != source_hw or self._resize_rows is None or self._resize_cols is None:
            self._source_hw = source_hw
            self._resize_rows = np.linspace(0, height - 1, BBF_OBS_SIZE, dtype=np.int32)
            self._resize_cols = np.linspace(0, width - 1, BBF_OBS_SIZE, dtype=np.int32)

        resized = gray[self._resize_rows][:, self._resize_cols]
        return np.asarray(resized, dtype=np.uint8)

    def _decision_raw_obs(self, fallback_raw: np.ndarray) -> np.ndarray:
        if self._interval_last_raw is None:
            return fallback_raw
        if self._interval_prev_raw is None:
            return self._interval_last_raw
        return np.maximum(self._interval_last_raw, self._interval_prev_raw)

    def _validate_action(self, action_idx: int) -> int:
        action_idx = int(action_idx)
        if action_idx < 0 or action_idx >= self._num_actions:
            raise ValueError(f"BBF produced out-of-bounds action {action_idx} for action_space={self._num_actions}")
        return action_idx

    def _choose_action(self, obs_u8: np.ndarray) -> int:
        action_idx = self._validate_action(int(self._agent.act(obs_u8)))
        self._held_action_idx = int(action_idx)
        self._has_held_action = True
        self._decision_steps += 1
        return int(action_idx)

    def _reset_interval(self) -> None:
        self._interval_reward = 0.0
        self._interval_count = 0
        self._interval_last_raw = None
        self._interval_prev_raw = None

    def _record_interval_raw(self, frame_raw: np.ndarray) -> None:
        if self._interval_last_raw is not None:
            self._interval_prev_raw = self._interval_last_raw
        self._interval_last_raw = np.asarray(frame_raw, dtype=np.uint8).copy()
        self._interval_count += 1

    def _finalize_interval(
        self,
        *,
        decision_obs: np.ndarray,
        next_action_obs: np.ndarray,
        terminated: bool,
        truncated: bool,
        episode_pulse: bool,
    ) -> None:
        if not self._has_held_action:
            self._choose_action(next_action_obs)
            return

        clipped_reward = self._clip_reward(self._interval_reward)
        step_truncated = bool(truncated or (episode_pulse and not terminated))

        train_stats = self._agent.step(
            decision_obs,
            float(clipped_reward),
            bool(terminated),
            bool(step_truncated),
            info={},
        )
        if isinstance(train_stats, dict):
            self._last_train_stats = dict(train_stats)
        self._transition_steps += 1

        # Select the next policy action immediately so the runner receives the
        # correct held action for the following raw frame.
        self._choose_action(next_action_obs)

    @staticmethod
    def _resolve_boundary_obs(boundary: Any, key: str) -> Optional[np.ndarray]:
        if not isinstance(boundary, Mapping):
            return None
        value = boundary.get(key)
        if value is None:
            return None
        return np.asarray(value)

    def frame(self, obs_rgb, reward, boundary) -> int:
        terminated, truncated, episode_pulse = self._parse_boundary(boundary)
        boundary_transition_obs = self._resolve_boundary_obs(boundary, "transition_obs_rgb")
        boundary_reset_obs = self._resolve_boundary_obs(boundary, "reset_obs_rgb")

        raw_transition_obs = boundary_transition_obs if boundary_transition_obs is not None else np.asarray(obs_rgb)
        if raw_transition_obs.dtype != np.uint8:
            raw_transition_obs = raw_transition_obs.astype(np.uint8)
        raw_reset_obs = boundary_reset_obs
        if raw_reset_obs is not None and raw_reset_obs.dtype != np.uint8:
            raw_reset_obs = raw_reset_obs.astype(np.uint8)
        self._raw_frames += 1

        if not self._has_held_action:
            self._choose_action(self._preprocess_frame(raw_transition_obs))

        self._interval_reward += float(reward)
        self._record_interval_raw(raw_transition_obs)

        should_finalize = bool(episode_pulse) or bool(self._interval_count >= BBF_ACTION_REPEAT)
        if should_finalize:
            decision_raw = self._decision_raw_obs(raw_transition_obs)
            decision_obs = self._preprocess_frame(decision_raw)
            if raw_reset_obs is not None:
                next_action_obs = self._preprocess_frame(raw_reset_obs)
            else:
                next_action_obs = decision_obs
            self._finalize_interval(
                decision_obs=decision_obs,
                next_action_obs=next_action_obs,
                terminated=bool(terminated),
                truncated=bool(truncated),
                episode_pulse=bool(episode_pulse),
            )
            self._reset_interval()

        return int(self._held_action_idx)

    def initial_action(self, obs_rgb) -> int:
        """Optional runner hook to choose the first episode action before stepping env."""
        raw_obs = np.asarray(obs_rgb)
        if raw_obs.dtype != np.uint8:
            raw_obs = raw_obs.astype(np.uint8)
        self._reset_interval()
        self._has_held_action = False
        return int(self._choose_action(self._preprocess_frame(raw_obs)))

    def step(self, obs_rgb, reward, terminated, truncated, info) -> int:
        del info
        return self.frame(
            obs_rgb=obs_rgb,
            reward=float(reward),
            boundary={
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "end_of_episode_pulse": bool(terminated) or bool(truncated),
            },
        )

    def _policy_action_for_eval(self, stacked_state: np.ndarray, rng: random.Random, epsilon: float) -> int:
        if rng.random() < float(epsilon):
            return int(rng.randrange(int(self._num_actions)))

        q_network = getattr(self._agent, "q_network", None)
        device = getattr(self._agent, "device", None)
        if q_network is not None and device is not None and hasattr(q_network, "compute_outputs"):
            try:
                import torch  # pylint: disable=import-outside-toplevel

                with torch.no_grad():
                    obs_t = torch.as_tensor(stacked_state, device=device, dtype=torch.uint8).unsqueeze(0)
                    outputs = q_network.compute_outputs(obs_t)
                    if isinstance(outputs, dict) and "q_values" in outputs:
                        q_values = outputs["q_values"]
                        action = int(torch.argmax(q_values, dim=1).item())
                        return self._validate_action(action)
            except Exception:
                pass

        # Fallback for non-standard/partial backends: use adapter-compatible act path.
        action = int(self._agent.act(stacked_state[-1]))
        return self._validate_action(action)

    def evaluate(
        self,
        env,
        *,
        episodes: int,
        epsilon: float,
        seed: int,
        clip_rewards: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Run no-learning single-game evaluation with native-style BBF decision cadence."""
        if int(episodes) <= 0:
            return None

        rng = random.Random(int(seed))
        returns: List[float] = []
        lengths: List[int] = []

        stacked = np.zeros((BBF_ACTION_REPEAT, BBF_OBS_SIZE, BBF_OBS_SIZE), dtype=np.uint8)
        obs = env.reset()
        first = self._preprocess_frame(np.asarray(obs, dtype=np.uint8))
        stacked = np.roll(stacked, -1, axis=0)
        stacked[-1] = first
        episode_return = 0.0
        episode_length = 0

        while len(returns) < int(episodes):
            action_idx = self._policy_action_for_eval(stacked, rng, float(epsilon))

            reward_sum = 0.0
            terminated = False
            truncated = False
            last_raw: Optional[np.ndarray] = None
            prev_raw: Optional[np.ndarray] = None

            for _ in range(BBF_ACTION_REPEAT):
                step = env.step(int(action_idx))
                reward_value = float(step.reward)
                if bool(clip_rewards):
                    reward_value = float(np.sign(reward_value))
                reward_sum += reward_value
                if last_raw is not None:
                    prev_raw = last_raw
                last_raw = np.asarray(step.obs_rgb, dtype=np.uint8)
                terminated = bool(step.terminated)
                truncated = bool(step.truncated)
                if terminated or truncated:
                    break

            if last_raw is None:
                break
            decision_raw = last_raw if prev_raw is None else np.maximum(last_raw, prev_raw)
            frame_u8 = self._preprocess_frame(decision_raw)
            stacked = np.roll(stacked, -1, axis=0)
            stacked[-1] = frame_u8

            episode_return += float(reward_sum)
            episode_length += 1

            if terminated or truncated:
                returns.append(float(episode_return))
                lengths.append(int(episode_length))
                episode_return = 0.0
                episode_length = 0
                obs = env.reset()
                stacked.fill(0)
                first = self._preprocess_frame(np.asarray(obs, dtype=np.uint8))
                stacked = np.roll(stacked, -1, axis=0)
                stacked[-1] = first

        if not returns:
            return None

        returns_np = np.asarray(returns, dtype=np.float32)
        lengths_np = np.asarray(lengths, dtype=np.float32)
        return {
            "episodes": int(len(returns)),
            "mean_return": float(np.mean(returns_np)),
            "std_return": float(np.std(returns_np)),
            "median_return": float(np.median(returns_np)),
            "min_return": float(np.min(returns_np)),
            "max_return": float(np.max(returns_np)),
            "mean_length": float(np.mean(lengths_np)),
            "episode_returns": [float(x) for x in returns],
            "episode_lengths": [int(x) for x in lengths],
            "epsilon": float(epsilon),
            "clip_rewards": bool(clip_rewards),
        }

    def get_config(self) -> Dict[str, Any]:
        return {
            **self._config.as_dict(),
            "seed": int(self._seed),
            "num_actions": int(self._num_actions),
            "total_frames": int(self._total_frames),
            "full_action_space": bool(self._agent_full_action_space),
            "parity_mode": bool(self._parity_mode),
            "action_space_mode": str(self._action_space_mode),
            "native_reset_semantics": bool(self._native_reset_semantics),
            "bbf_action_repeat": int(BBF_ACTION_REPEAT),
            "bbf_obs_size": int(BBF_OBS_SIZE),
        }

    def get_stats(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            "raw_frames": int(self._raw_frames),
            "decision_steps": int(self._decision_steps),
            "transition_steps": int(self._transition_steps),
            "last_action_idx": int(self._held_action_idx),
            "bbf_parity_mode": bool(self._parity_mode),
            "bbf_native_reset_semantics": bool(self._native_reset_semantics),
            "action_space_mode": str(self._action_space_mode),
            "full_action_space": bool(self._agent_full_action_space),
            "learning_starts": int(self._config.learning_starts),
            "buffer_size": int(self._config.buffer_size),
        }

        replay_buffer = getattr(self._agent, "replay_buffer", None)
        replay_add_count: Optional[int] = None
        replay_size: Optional[int] = None
        if replay_buffer is not None:
            add_count = getattr(replay_buffer, "add_count", None)
            if isinstance(add_count, (int, np.integer)):
                replay_add_count = int(add_count)
            size_fn = getattr(replay_buffer, "num_elements", None)
            if callable(size_fn):
                try:
                    size_value = size_fn()
                    if isinstance(size_value, (int, np.integer)):
                        replay_size = int(size_value)
                except Exception:  # pragma: no cover - defensive
                    replay_size = None
            if replay_size is None:
                try:
                    replay_size = int(len(replay_buffer))
                except Exception:  # pragma: no cover - defensive
                    replay_size = None
        if replay_add_count is not None:
            stats["replay_add_count"] = int(replay_add_count)
        if replay_size is not None:
            stats["replay_size"] = int(replay_size)

        for key in ("training_steps", "grad_steps", "global_step"):
            value = getattr(self._agent, key, None)
            if isinstance(value, (int, np.integer)):
                mapped = "train_steps" if key == "training_steps" else str(key)
                stats[mapped] = int(value)
        for key in ("learning_ready_step", "update_period"):
            value = getattr(self._agent, key, None)
            if isinstance(value, (int, np.integer)):
                stats[str(key)] = int(value)

        if self._last_train_stats:
            metric_map = {
                "loss": "last_train_loss",
                "spr_loss": "last_train_spr_loss",
                "avg_q": "last_train_avg_q",
                "gamma": "last_train_gamma",
            }
            for src, dst in metric_map.items():
                if src in self._last_train_stats:
                    value = _to_json_scalar(self._last_train_stats[src])
                    if isinstance(value, (int, float, bool, str)) or value is None:
                        stats[dst] = value

        train_steps_value = stats.get("train_steps")
        grad_steps_value = stats.get("grad_steps")
        learning_ready_step = stats.get("learning_ready_step")
        update_period = stats.get("update_period")
        can_train_gate = None
        if isinstance(replay_add_count, int) and isinstance(train_steps_value, int):
            ready_step = (
                int(learning_ready_step)
                if isinstance(learning_ready_step, int)
                else int(self._config.learning_starts)
            )
            period = int(update_period) if isinstance(update_period, int) and int(update_period) > 0 else 1
            can_train_gate = bool(
                int(replay_add_count) > int(self._config.learning_starts)
                and int(train_steps_value) >= int(ready_step)
                and (int(train_steps_value) % int(period) == 0)
            )

        # "training" should indicate BBF updates are active, not just post-warmup replay fill.
        if isinstance(grad_steps_value, int):
            stats["phase"] = "training" if int(grad_steps_value) > 0 else "warmup"
        elif isinstance(can_train_gate, bool):
            stats["phase"] = "training" if bool(can_train_gate) else "warmup"

        get_stats_fn = getattr(self._agent, "get_stats", None)
        if callable(get_stats_fn):
            try:
                payload = get_stats_fn()
                if isinstance(payload, dict):
                    for key, value in payload.items():
                        if key in stats:
                            continue
                        scalar = _to_json_scalar(value)
                        if isinstance(scalar, (int, float, bool, str)) or scalar is None:
                            stats[str(key)] = scalar
            except Exception:  # pragma: no cover - defensive
                pass
        return stats
