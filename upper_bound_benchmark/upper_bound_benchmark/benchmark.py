import json
import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .env import make_atari_env, FrameCounterWrapper, RealTimePacingWrapper, ActionLatencyWrapper


DEFAULT_GAMES: List[str] = [
    "ALE/Atlantis-v5",
    "ALE/BattleZone-v5",
    "ALE/Centipede-v5",
    "ALE/Defender-v5",
    "ALE/Krull-v5",
    "ALE/MsPacman-v5",
    "ALE/Qbert-v5",
    "ALE/UpNDown-v5",
]


@dataclass
class BenchmarkConfig:
    cycles: int = 3
    frames_per_game: int = 400_000
    sticky_prob: float = 0.25
    full_action_space: bool = True
    target_fps: float = 60.0
    realtime: bool = True
    latency_frames: int = 0
    seed: int = 0
    render: bool = False
    output_dir: str = "results"


class BenchmarkRunner:
    def __init__(self, games: List[str], config: BenchmarkConfig):
        self.games = list(games)
        self.cfg = config
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        self.rng = np.random.default_rng(self.cfg.seed)

    def _build_env(self, game_id: str, game_seed: int):
        render_mode = "human" if self.cfg.render else None
        env = make_atari_env(
            game_id,
            seed=game_seed,
            frameskip=1,
            repeat_action_probability=self.cfg.sticky_prob,
            full_action_space=self.cfg.full_action_space,
            render_mode=render_mode,
        )
        env = FrameCounterWrapper(env)
        if self.cfg.latency_frames > 0:
            env = ActionLatencyWrapper(env, latency_frames=self.cfg.latency_frames)
        if self.cfg.realtime:
            env = RealTimePacingWrapper(env, target_fps=self.cfg.target_fps)
        return env

    def run(self, agent) -> Dict[str, Dict[str, float]]:
        results: Dict[str, Dict[str, float]] = {}
        per_cycle_scores: Dict[int, Dict[str, float]] = {i: {} for i in range(1, self.cfg.cycles + 1)}

        for cycle in range(1, self.cfg.cycles + 1):
            for game_index, game_id in enumerate(self.games):
                print(f"\nCycle {cycle}/{self.cfg.cycles}, Game: {game_id}")
                game_seed = self.cfg.seed + 1000 * cycle + game_index
                env = self._build_env(game_id, game_seed)
                obs, info = env.reset()
                frames_target = self.cfg.frames_per_game
                total_return_this_cycle = 0.0
                episode_return = 0.0
                frames_used = 0

                agent.begin_task(game_id=game_id, cycle=cycle, action_space=env.action_space)
                print(f"  Target: {frames_target} frames, Seed: {game_seed}")

                while frames_used < frames_target:
                    action = agent.act(obs, env.action_space)
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    frames_used += 1
                    agent.observe(obs, action, reward, next_obs, terminated, truncated, info)
                    episode_return += float(reward)

                    done = terminated or truncated
                    if done:
                        agent.end_episode(episode_return)
                        total_return_this_cycle += episode_return
                        episode_return = 0.0
                        obs, info = env.reset()
                    else:
                        obs = next_obs
                    
                    # Progress logging
                    if frames_used % 10000 == 0:
                        print(f"  Progress: {frames_used}/{frames_target} frames ({100*frames_used/frames_target:.1f}%), Return so far: {total_return_this_cycle:.1f}")

                # If we ended mid-episode, count partial return
                if episode_return != 0.0:
                    total_return_this_cycle += episode_return
                    agent.end_episode(episode_return)

                env.close()
                per_cycle_scores[cycle][game_id] = total_return_this_cycle

        # Final scoring is sum of last cycle across all games
        final_cycle = self.cfg.cycles
        total_final = float(sum(per_cycle_scores[final_cycle].values()))
        results["final_cycle_scores"] = per_cycle_scores[final_cycle]
        results["final_total"] = total_final
        results["per_cycle_scores"] = per_cycle_scores
        results["config"] = {
            "cycles": self.cfg.cycles,
            "frames_per_game": self.cfg.frames_per_game,
            "sticky_prob": self.cfg.sticky_prob,
            "full_action_space": self.cfg.full_action_space,
            "target_fps": self.cfg.target_fps,
            "realtime": self.cfg.realtime,
            "latency_frames": self.cfg.latency_frames,
            "seed": self.cfg.seed,
            "render": self.cfg.render,
        }

        out_path = os.path.join(self.cfg.output_dir, "results.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

        return results
