import numpy as np

from aasma import Agent

class RandomPrey(Agent):
    def __init__(self, n_actions: int, prey_id):
        super(RandomPrey, self).__init__("Random Prey")
        self.n_actions = n_actions
        self.prey_id = prey_id

    def action(self) -> int:
        return np.random.randint(self.n_actions)
    
    def next(self, observation, action, next_observation, reward, terminal, info):
        # Not a learning agent
        pass