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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import gymnasium as gym
from ale_py import Action, ALEInterface, LoggerMode, roms
from cleanrl_utils.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)


class StickyActionEnv(gym.Wrapper):
    """
    Implements sticky actions as described in Machado et al. (2018).
    With probability `sticky_prob`, the previous action is repeated instead of the new action.
    """
    def __init__(self, env: gym.Env, sticky_prob: float = 0.25):
        super().__init__(env)
        self.sticky_prob = sticky_prob
        self.last_action = 0

    def reset(self, **kwargs):
        self.last_action = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        if self.env.unwrapped.np_random.random() < self.sticky_prob:
            action = self.last_action
        self.last_action = action
        return self.env.step(action)


class CanonicalActionWrapper(gym.ActionWrapper):
    """
    Forces the environment to accept the full 18-action canonical set.
    Maps canonical actions to the environment's legal actions.
    If a canonical action is not supported by the game, it maps to NOOP (0).
    
    This allows an agent with a fixed 18-output head to play games with 
    smaller action spaces (e.g. Breakout) without crashing.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        # 1. Define the Canonical 18 actions (Must match BenchmarkRunner._canonical_action_set)
        self.canonical_actions = [
            Action.NOOP, Action.FIRE, Action.UP, Action.RIGHT, Action.LEFT, Action.DOWN,
            Action.UPRIGHT, Action.UPLEFT, Action.DOWNRIGHT, Action.DOWNLEFT,
            Action.UPFIRE, Action.RIGHTFIRE, Action.LEFTFIRE, Action.DOWNFIRE,
            Action.UPRIGHTFIRE, Action.UPLEFTFIRE, Action.DOWNRIGHTFIRE, Action.DOWNLEFTFIRE
        ]
        
        # 2. Map Canonical Index (0-17) -> Native Env Index (0-N)
        self.action_map = {}
        
        # get_action_meanings() returns uppercase strings like ['NOOP', 'FIRE', 'RIGHT', ...]
        native_meanings = env.unwrapped.get_action_meanings()
        
        for c_idx, c_enum in enumerate(self.canonical_actions):
            # c_enum.name gives "UP", "FIRE", etc.
            c_name = c_enum.name 
            
            try:
                # Find the index of this action name in the specific game's list
                native_idx = native_meanings.index(c_name)
                self.action_map[c_idx] = native_idx
            except ValueError:
                # The game does not support this action (e.g. UPFIRE in Breakout)
                # Map to NOOP (usually index 0)
                self.action_map[c_idx] = 0 

        # 3. Expose the full 18-action space to the agent
        self.action_space = gym.spaces.Discrete(18)

    def action(self, action):
        # Remap the agent's selection (0-17) to the game's supported selection (0-N)
        return self.action_map.get(int(action), 0)


def rom_name_to_attr(rom_name: str) -> str:
    parts = [part for part in rom_name.strip().replace("-", "_").split("_") if part]
    if not parts:
        raise ValueError(f"Cannot derive ROM attribute from '{rom_name}'")
    return "".join(part.capitalize() for part in parts)


def resolve_rom_path(rom_name: str):
    attr = rom_name_to_attr(rom_name)
    if not hasattr(roms, attr):
        available = ", ".join(sorted(roms.__dir__()))
        raise AttributeError(f"No ROM named '{attr}'. Supported ROMs: {available}")
    return getattr(roms, attr)


def rom_name_to_env_id(rom_name: str) -> str:
    sanitized = rom_name.strip().lower().replace("-", "_")
    parts = [part for part in sanitized.split("_") if part]
    if not parts:
        raise ValueError(f"Cannot derive Gym env id from ROM name '{rom_name}'")
    camel = "".join(part.capitalize() for part in parts)
    return f"{camel}NoFrameskip-v4"


@dataclass
class GameSpec:
    """Describe one game run within a cycle."""

    name: str
    frame_budget: int
    sticky_prob: float = 0.0
    delay_frames: int = 0
    seed: Optional[int] = None
    params: Dict[str, Any] = field(default_factory=dict)
    backend: str = "ale"
    env_id: Optional[str] = None


@dataclass
class CycleConfig:
    """Describe the ordered list of games to play in a cycle."""

    cycle_index: int
    games: List[GameSpec]


@dataclass
class BenchmarkConfig:
    """Full benchmark specification across multiple cycles."""

    cycles: List[CycleConfig]
    description: Optional[str] = None

    @property
    def total_frames(self) -> int:
        return sum(game.frame_budget for cycle in self.cycles for game in cycle.games)


@dataclass
class GameResult:
    """Metrics captured for a single game execution."""

    cycle_index: int
    game_index: int
    spec: GameSpec
    name: str
    frame_offset: int
    frame_budget: int
    episode_scores: List[float]
    episode_end: List[int]
    episode_graph: Any
    parms_graph: Any


@dataclass
class EnvironmentHandle:
    """Encapsulate the environment reference passed to a frame runner."""

    backend: str
    spec: GameSpec
    frames_per_step: int = 1
    ale: Optional[ALEInterface] = None
    gym: Optional[gym.Env] = None
    action_set: Optional[List[int]] = None


@dataclass
class FrameRunnerContext:
    """Run-specific metadata shared with the frame runner."""

    name: str
    data_dir: str
    rank: int
    average_frames: int
    max_frames_without_reward: int
    lives_as_episodes: int
    save_incremental_models: bool
    last_model_save: int
    frame_budget: int
    frame_offset: int
    graph_total_frames: int
    delay_frames: int


@dataclass
class FrameRunnerResult:
    """Structured result returned by a frame runner."""

    last_model_save: int
    episode_scores: List[float]
    episode_end: List[int]
    episode_graph: Any
    parms_graph: Any


class BenchmarkRunner:
    """Continual benchmark harness coordinating cycles and game runs."""

    def __init__(
        self,
        agent,
        config: BenchmarkConfig,
        *,
        frame_runner: Callable[..., Any],
        data_dir: str,
        rank: int = 0,
        default_seed: Optional[int] = None,
        reduce_action_set: int = 0,
        use_canonical_full_actions: bool = False,
        average_frames: int = 100_000,
        max_frames_without_reward: int = 18_000,
        lives_as_episodes: int = 1,
        save_incremental_models: bool = False,
    ):
        self.agent = agent
        self.config = config
        self._frame_runner = frame_runner
        self.data_dir = data_dir
        self.rank = rank
        self.default_seed = 0 if default_seed is None else default_seed
        self.reduce_action_set = reduce_action_set
        self.use_canonical_full_actions = use_canonical_full_actions
        self.average_frames = average_frames
        self.max_frames_without_reward = max_frames_without_reward
        self.lives_as_episodes = lives_as_episodes
        self.save_incremental_models = save_incremental_models

        self.last_model_save = -1
        self.frame_offset = 0
        self.results: List[GameResult] = []

    def run(self) -> List[GameResult]:
        for cycle in self.config.cycles:
            for game_index, spec in enumerate(cycle.games):
                result = self._run_single_game(cycle.cycle_index, game_index, spec)
                self.results.append(result)
                self.frame_offset += spec.frame_budget
        return self.results

    def _run_single_game(self, cycle_index: int, game_index: int, spec: GameSpec) -> GameResult:
        handle = self._create_environment(spec)
        if handle.backend == 'ale':
            if handle.ale is None:
                raise ValueError('ALE backend selected but no ALEInterface available.')
            action_set = self._select_action_set(handle.ale, spec)
            handle.action_set = action_set
            expected_actions = getattr(self.agent, 'num_actions', None)
            if expected_actions is None:
                raise ValueError('Agent must define num_actions when using ALE backend.')
            if len(action_set) != expected_actions:
                raise ValueError(
                    f'Action set size {len(action_set)} for game {spec.name} does not match agent.num_actions {expected_actions}'
                )
        run_name = self._make_run_name(cycle_index, game_index, spec)

        context = FrameRunnerContext(
            name=run_name,
            data_dir=self.data_dir,
            rank=self.rank,
            average_frames=self.average_frames,
            max_frames_without_reward=spec.params.get('max_frames_without_reward', self.max_frames_without_reward),
            lives_as_episodes=spec.params.get('lives_as_episodes', self.lives_as_episodes),
            save_incremental_models=self.save_incremental_models,
            last_model_save=self.last_model_save,
            frame_budget=spec.frame_budget,
            frame_offset=self.frame_offset,
            graph_total_frames=self.config.total_frames,
            delay_frames=spec.delay_frames,
        )

        try:
            runner_result = self._frame_runner(self.agent, handle, context=context)
        finally:
            if handle.backend == 'gym' and handle.gym is not None:
                handle.gym.close()

        if isinstance(runner_result, FrameRunnerResult):
            result_payload = runner_result
        else:
            (
                last_model_save,
                episode_scores,
                episode_end,
                episode_graph,
                parms_graph,
            ) = runner_result
            result_payload = FrameRunnerResult(last_model_save, episode_scores, episode_end, episode_graph, parms_graph)

        self.last_model_save = result_payload.last_model_save
        episode_scores = result_payload.episode_scores
        episode_end = result_payload.episode_end
        episode_graph = result_payload.episode_graph
        parms_graph = result_payload.parms_graph

        result = GameResult(
            cycle_index=cycle_index,
            game_index=game_index,
            spec=spec,
            name=run_name,
            frame_offset=self.frame_offset,
            frame_budget=spec.frame_budget,
            episode_scores=episode_scores,
            episode_end=episode_end,
            episode_graph=episode_graph,
            parms_graph=parms_graph,
        )

        return result

    def _create_environment(self, spec: GameSpec) -> EnvironmentHandle:
        backend = spec.params.get('env_backend', spec.backend).lower()

        if backend == 'gym':
            env_seed = spec.seed if spec.seed is not None else self.default_seed
            env_id = spec.env_id or spec.params.get('env_id') or rom_name_to_env_id(spec.name)
            env = self._make_gym_env(env_id, env_seed, spec)
            frame_skip = int(spec.params.get('frame_skip', 4))
            frames_per_step = max(1, frame_skip)
            return EnvironmentHandle(
                backend='gym',
                spec=spec,
                frames_per_step=frames_per_step,
                gym=env,
            )

        if backend != 'ale':
            raise ValueError(f"Unsupported environment backend '{backend}' for game {spec.name}")

        ale = ALEInterface()
        ale.setLoggerMode(LoggerMode.Error)
        ale.setInt('random_seed', spec.seed if spec.seed is not None else self.default_seed)
        ale.setFloat('repeat_action_probability', spec.sticky_prob)

        for key, value in spec.params.get('ale_ints', {}).items():
            ale.setInt(key, int(value))
        for key, value in spec.params.get('ale_floats', {}).items():
            ale.setFloat(key, float(value))

        rom_path = resolve_rom_path(spec.name)
        ale.loadROM(rom_path)
        ale.reset_game()
        frames_per_step = max(1, int(spec.params.get('frame_skip', 1)))
        return EnvironmentHandle(backend='ale', spec=spec, frames_per_step=frames_per_step, ale=ale)

    def _select_action_set(self, ale: ALEInterface, spec: GameSpec) -> List[int]:
        if self.use_canonical_full_actions or spec.params.get('use_canonical_full_actions'):
            legal_actions = list(ale.getLegalActionSet())
            legal = set(legal_actions)
            canonical = self._canonical_action_set()
            fallback = canonical[0]
            if fallback not in legal:
                fallback = legal_actions[0]
            action_set = [action if action in legal else fallback for action in canonical]
        else:
            action_set = self._select_reduced_action_set(ale, spec)

        return action_set

    def _make_gym_env(self, env_id: str, seed: int, spec: GameSpec) -> gym.Env:
        env = gym.make(env_id)
        
        # --- FIX START: Apply Canonical Action Wrapper ---
        # Forces the environment to accept 18 actions if the agent is configured that way
        if self.use_canonical_full_actions or spec.params.get('use_canonical_full_actions'):
            env = CanonicalActionWrapper(env)
        # --- FIX END ---

        # Add sticky actions wrapper if requested
        if spec.sticky_prob > 0.0:
            env = StickyActionEnv(env, sticky_prob=spec.sticky_prob)

        env = gym.wrappers.RecordEpisodeStatistics(env)

        noop_max = int(spec.params.get('noop_max', 30))
        frame_skip = int(spec.params.get('frame_skip', 4))
        frame_stack = int(spec.params.get('frame_stack', 4))
        resize_shape = tuple(spec.params.get('resize_shape', (84, 84)))
        grayscale = bool(spec.params.get('grayscale', True))

        env = NoopResetEnv(env, noop_max=noop_max)
        env = MaxAndSkipEnv(env, skip=frame_skip)
        env = EpisodicLifeEnv(env)
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, resize_shape)
        if grayscale:
            env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, frame_stack)

        env.action_space.seed(seed)
        env.reset(seed=seed)
        return env

    def _select_reduced_action_set(self, ale: ALEInterface, spec: GameSpec) -> List[int]:
        reduce_action_set = spec.params.get('reduce_action_set', self.reduce_action_set)
        if reduce_action_set == 0:
            action_set = ale.getLegalActionSet()
        elif reduce_action_set == 2 and spec.name in {'ms_pacman', 'qbert'}:
            action_set = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
        else:
            action_set = ale.getMinimalActionSet()

        return action_set

    def _make_run_name(self, cycle_index: int, game_index: int, spec: GameSpec) -> str:
        return f'cycle{cycle_index:02d}_game{game_index:02d}_{spec.name}_delay{spec.delay_frames}'

    @staticmethod
    def _canonical_action_set() -> List[int]:
        return [
            Action.NOOP,
            Action.FIRE,
            Action.UP,
            Action.RIGHT,
            Action.LEFT,
            Action.DOWN,
            Action.UPRIGHT,
            Action.UPLEFT,
            Action.DOWNRIGHT,
            Action.DOWNLEFT,
            Action.UPFIRE,
            Action.RIGHTFIRE,
            Action.LEFTFIRE,
            Action.DOWNFIRE,
            Action.UPRIGHTFIRE,
            Action.UPLEFTFIRE,
            Action.DOWNRIGHTFIRE,
            Action.DOWNLEFTFIRE,
        ]