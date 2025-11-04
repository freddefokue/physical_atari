from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ale_py import Action, ALEInterface, LoggerMode, roms


@dataclass
class GameSpec:
    """Describe one game run within a cycle."""

    name: str
    frame_budget: int
    sticky_prob: float = 0.0
    delay_frames: int = 0
    seed: Optional[int] = None
    params: Dict[str, Any] = field(default_factory=dict)


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
        ale = self._create_environment(spec)
        action_set = self._select_action_set(ale, spec)
        run_name = self._make_run_name(cycle_index, game_index, spec)

        (
            self.last_model_save,
            episode_scores,
            episode_end,
            episode_graph,
            parms_graph,
        ) = self._frame_runner(
            self.agent,
            ale,
            action_set,
            name=run_name,
            data_dir=self.data_dir,
            rank=self.rank,
            delay_frames=spec.delay_frames,
            average_frames=self.average_frames,
            max_frames_without_reward=spec.params.get(
                'max_frames_without_reward', self.max_frames_without_reward
            ),
            lives_as_episodes=spec.params.get('lives_as_episodes', self.lives_as_episodes),
            save_incremental_models=self.save_incremental_models,
            last_model_save=self.last_model_save,
            frame_budget=spec.frame_budget,
            frame_offset=self.frame_offset,
            graph_total_frames=self.config.total_frames,
        )

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

    def _create_environment(self, spec: GameSpec) -> ALEInterface:
        ale = ALEInterface()
        ale.setLoggerMode(LoggerMode.Error)
        ale.setInt('random_seed', spec.seed if spec.seed is not None else self.default_seed)
        ale.setFloat('repeat_action_probability', spec.sticky_prob)

        for key, value in spec.params.get('ale_ints', {}).items():
            ale.setInt(key, int(value))
        for key, value in spec.params.get('ale_floats', {}).items():
            ale.setFloat(key, float(value))

        rom_path = roms.get_rom_path(spec.name)
        ale.loadROM(rom_path)
        ale.reset_game()
        return ale

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

        if len(action_set) != self.agent.num_actions:
            raise ValueError(
                f'Action set size {len(action_set)} for game {spec.name} does not match agent.num_actions {self.agent.num_actions}'
            )

        return action_set

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


