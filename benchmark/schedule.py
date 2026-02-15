"""Deterministic multi-game visit scheduling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Sequence

import numpy as np


@dataclass(frozen=True)
class ScheduleVisit:
    """One scheduled game visit in the continual stream."""

    visit_idx: int
    cycle_idx: int
    game_id: str
    visit_frames: int


@dataclass(frozen=True)
class ScheduleConfig:
    """Inputs for deterministic schedule construction."""

    games: Sequence[str]
    base_visit_frames: int
    num_cycles: int
    seed: int = 0
    jitter_pct: float = 0.0
    min_visit_frames: int = 1

    def __post_init__(self) -> None:
        if not self.games:
            raise ValueError("games must not be empty")
        if self.base_visit_frames <= 0:
            raise ValueError("base_visit_frames must be > 0")
        if self.num_cycles <= 0:
            raise ValueError("num_cycles must be > 0")
        if self.jitter_pct < 0.0 or self.jitter_pct >= 1.0:
            raise ValueError("jitter_pct must be in [0.0, 1.0)")
        if self.min_visit_frames <= 0:
            raise ValueError("min_visit_frames must be > 0")


class Schedule:
    """Reproducible visit schedule (order + jittered lengths)."""

    def __init__(self, config: ScheduleConfig) -> None:
        self.config = config
        self._visits = self._build_visits(config)

    @staticmethod
    def _jittered_frames(
        base_visit_frames: int,
        jitter_pct: float,
        min_visit_frames: int,
        rng: np.random.Generator,
    ) -> int:
        if jitter_pct <= 0.0:
            return max(int(min_visit_frames), int(base_visit_frames))
        scale = 1.0 + float(rng.uniform(-jitter_pct, jitter_pct))
        jittered = int(round(base_visit_frames * scale))
        return max(int(min_visit_frames), int(jittered))

    @classmethod
    def _build_visits(cls, config: ScheduleConfig) -> List[ScheduleVisit]:
        rng = np.random.default_rng(int(config.seed))
        games = [str(game) for game in config.games]
        visits: List[ScheduleVisit] = []

        visit_idx = 0
        for cycle_idx in range(config.num_cycles):
            cycle_order = list(games)
            rng.shuffle(cycle_order)
            for game_id in cycle_order:
                visit_frames = cls._jittered_frames(
                    config.base_visit_frames,
                    config.jitter_pct,
                    config.min_visit_frames,
                    rng,
                )
                visits.append(
                    ScheduleVisit(
                        visit_idx=visit_idx,
                        cycle_idx=cycle_idx,
                        game_id=game_id,
                        visit_frames=visit_frames,
                    )
                )
                visit_idx += 1

        return visits

    @property
    def visits(self) -> List[ScheduleVisit]:
        return list(self._visits)

    @property
    def total_frames(self) -> int:
        return int(sum(visit.visit_frames for visit in self._visits))

    def __iter__(self) -> Iterator[ScheduleVisit]:
        return iter(self._visits)

    def __len__(self) -> int:
        return len(self._visits)

    def as_records(self) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for visit in self._visits:
            records.append(
                {
                    "visit_idx": int(visit.visit_idx),
                    "cycle_idx": int(visit.cycle_idx),
                    "game_id": str(visit.game_id),
                    "visit_frames": int(visit.visit_frames),
                }
            )
        return records


def schedule_records(visits: Iterable[ScheduleVisit]) -> List[Dict[str, Any]]:
    """Convert visit objects to JSON-serializable records."""

    return [
        {
            "visit_idx": int(visit.visit_idx),
            "cycle_idx": int(visit.cycle_idx),
            "game_id": str(visit.game_id),
            "visit_frames": int(visit.visit_frames),
        }
        for visit in visits
    ]
