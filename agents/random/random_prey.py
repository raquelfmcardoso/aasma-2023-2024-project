import numpy as np

from aasma import Agent

class RandomPrey(Agent):
    def __init__(self, n_actions: int):
        super(RandomPrey, self).__init__("Random Prey")
        self.n_actions = n_actions

    def action(self) -> int:
        return np.random.randint(self.n_actions)
    
    def next(self, observation, action, next_observation, reward, terminal, info):
        # Not a learning agent
        pass