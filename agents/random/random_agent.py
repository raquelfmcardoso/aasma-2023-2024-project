import numpy as np

from aasma import Agent


class RandomAgent(Agent):
    def __init__(self, n_actions: int, agent_id):
        super(RandomAgent, self).__init__("Random Agent")
        self.n_actions = n_actions
        self.agent_id = agent_id

    def action(self) -> int:
        return np.random.randint(self.n_actions)
    
    def run(self) -> int:
        return self.action()
    
    def next(self, observation, action, next_observation, reward, terminal, info):
        # Not a learning agent
        pass


