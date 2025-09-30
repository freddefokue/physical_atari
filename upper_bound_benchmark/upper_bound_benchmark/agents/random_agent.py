from .base import Agent


class RandomAgent(Agent):
    def act(self, obs, action_space) -> int:
        return action_space.sample()
