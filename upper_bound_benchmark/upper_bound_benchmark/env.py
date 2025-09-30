import time
import collections
from typing import Optional

import gymnasium as gym
import numpy as np

# Import ale_py to register ALE environments with gymnasium
import ale_py
gym.register_envs(ale_py)


class FrameCounterWrapper(gym.Wrapper):
    """Counts environment steps as frames (with frameskip=1)."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.frame_count = 0

    def reset(self, *args, **kwargs):
        self.frame_count = 0
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frame_count += 1
        info = dict(info)
        info["frame_count"] = self.frame_count
        return obs, reward, terminated, truncated, info


class RealTimePacingWrapper(gym.Wrapper):
    """Paces environment to target FPS in wall time."""

    def __init__(self, env: gym.Env, target_fps: float = 60.0):
        super().__init__(env)
        self.step_duration = 1.0 / float(target_fps)
        self._last_step_end_time: Optional[float] = None

    def reset(self, *args, **kwargs):
        self._last_step_end_time = time.perf_counter()
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        start = time.perf_counter()
        # Enforce pacing based on last step end -> next step start
        if self._last_step_end_time is not None:
            elapsed = start - self._last_step_end_time
            remaining = self.step_duration - elapsed
            if remaining > 0:
                time.sleep(remaining)
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_step_end_time = time.perf_counter()
        return obs, reward, terminated, truncated, info


class ActionLatencyWrapper(gym.Wrapper):
    """Delays the applied action by N frames (control latency).

    The action passed to `step` is enqueued, and the environment receives the
    oldest queued action. If the queue hasn't filled yet, NOOP is used.
    """

    def __init__(self, env: gym.Env, latency_frames: int = 0):
        super().__init__(env)
        assert latency_frames >= 0
        self.latency_frames = int(latency_frames)
        self._queue = collections.deque(maxlen=self.latency_frames + 1)
        self._noop_action = self._infer_noop_action()

    def _infer_noop_action(self) -> int:
        # Try to find NOOP in action meanings; default to 0
        try:
            meanings = self.env.unwrapped.get_action_meanings()
            for idx, meaning in enumerate(meanings):
                if meaning.upper() == "NOOP":
                    return idx
        except Exception:
            pass
        return 0

    def reset(self, *args, **kwargs):
        self._queue.clear()
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        if self.latency_frames == 0:
            return self.env.step(action)
        # Enqueue the requested action; apply the oldest
        self._queue.append(action)
        if len(self._queue) <= self.latency_frames:
            applied = self._noop_action
        else:
            applied = self._queue.popleft()
        return self.env.step(applied)


def make_atari_env(
    game_id: str,
    seed: Optional[int] = None,
    frameskip: int = 1,
    repeat_action_probability: float = 0.25,
    full_action_space: bool = True,
    render_mode: Optional[str] = None,
) -> gym.Env:
    """Factory for ALE Atari environments with required settings.

    - frameskip=1 so one env.step equals one frame
    - sticky actions via `repeat_action_probability`
    - full_action_space=True
    """
    assert frameskip == 1, "Benchmark requires frameskip=1"

    env = gym.make(
        game_id,
        render_mode=render_mode,
        frameskip=frameskip,
        repeat_action_probability=repeat_action_probability,
        full_action_space=full_action_space,
    )
    # Seed if provided
    if seed is not None:
        try:
            env.reset(seed=seed)
        except TypeError:
            pass
    return env
