import numpy as np
import math
from aasma import Agent
from gym import Env
from scipy.spatial.distance import cityblock

N_ACTIONS = 5
DOWN, LEFT, UP, RIGHT, STAY = range(N_ACTIONS)
ROLES = ['alpha', 'beta', 'gamma', 'delta']


class BdiAgent(Agent):
    def __init__(self, agent_id, role):
        # Initialize agent's beliefs, desires, and intentions
        self.agent_id = agent_id
        self.n_actions = N_ACTIONS
        self.role = role
        self.beliefs = {}
        self.desires = []
        self.intentions = []

    def perceive(self, environment):
        # Update agent's beliefs based on the current environment
        agent_position = self.observation[self.agent_id][0][0], self.observation[self.agent_id][0][1]
        prey_positions = self.observation[self.agent_id][1]
        wall_positions = self.observation[self.agent_id][2]
        self.beliefs['agent_position'] = agent_position
        self.beliefs['prey_positions'] = prey_positions
        self.beliefs['wall_positions'] = wall_positions
        #self.beliefs['near_agents'] = near_agents
        #self.beliefs['near_preys'] = near_preys 
        pass

    def update_desires(self):
        # Update agent's desires based on its beliefs
        closest_prey = self.closest_prey(self.beliefs['agent_position'], self.beliefs['prey_positions'])
        self.desires.append(closest_prey)
        pass

    def deliberation(self):
        # Select intentions based on agent's desires and beliefs
        pass

    def action(self):
        # Execute the selected intentions
        pass

    def run(self, environment):
        # Run the BDI agent's decision-making loop
        while True:
            self.perceive(environment)
            self.update_desires()
            self.deliberation()
            self.action()
            
    def closest_prey(self, agent_position, prey_positions):
        """
        Given the positions of an agent and a sequence of positions of all prey,
        returns the positions of the closest prey.
        If there are no preys, None is returned instead
        """
        min = math.inf
        closest_prey_position = None
        n_preys = len(prey_positions)
        for p in range(n_preys):
            prey_position = prey_positions[p][0], prey_positions[p][1]
            distance = cityblock(agent_position, prey_position)
            if distance < min:
                min = distance
                closest_prey_position = prey_position
        return closest_prey_position