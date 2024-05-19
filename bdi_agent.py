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
        self.desires = {}
        self.intentions = []

    def perceive(self, environment):
        # Update agent's beliefs based on the current environment
        agent_position = self.observation[self.agent_id][0][1]
        agent_positions = self.observation[self.agent_id][1]
        prey_positions = self.observation[self.agent_id][2]
        wall_positions = self.observation[self.agent_id][3]
        self.beliefs['agent_position'] = agent_position
        self.beliefs['agent_positions'] = agent_positions
        self.beliefs['prey_positions'] = prey_positions
        self.beliefs['wall_positions'] = wall_positions

        # We get the complete observations of agents we can see
        agent_id_absolute_obs = {}
        # We only get the amount of preys an agent we can't see can see
        agent_id_relative_obs = {}

        agent_id_absolute_obs[self.agent_id] = self.observation[self.agent_id] # add our own observations
        for agent in agent_positions:
            agent_id_absolute_obs[agent[0]] = self.observation[agent[0]] # {agent_id} : {self.observations[agent_id]}

        for agent_observation in self.observation:
            if (agent_observation[0][0] not in agent_id_absolute_obs): # if we can't see the agent
                agent_id_relative_obs[tuple(agent_observation[0][1])] = len(self.observation[agent_observation[0][0]][2]) # {agent_position} : {amount_of_preys}

        self.beliefs['absolute_obs'] = agent_id_absolute_obs
        self.beliefs['relative_obs'] = agent_id_relative_obs

    def update_desires(self):
        # Update agent's desires based on its beliefs
        def get_closest_agent_relative():
            if len(self.beliefs['relative_obs']) > 0:
                agents = []
                for agent_pos, num_preys in self.beliefs['relative_obs'].items():
                    agents.append(num_preys, cityblock(self.beliefs['agent_position'], agent_pos))
                agents.sort(key=lambda x: x[1])
                for agent in agents:
                    if agent[0] >= 2:
                        self.desires['closest_prey'] = agent[2]
                        break
                if 'closest_prey' not in self.desires:
                    for agent in agents:
                        if agent[0] == 1:
                            self.desires['closest_prey'] = agent[2]
                            break
                if 'closest_prey' not in self.desires:
                    #Move randomly
                    pass
            else: 
                #Move randomly
                pass

        if len(self.beliefs['prey_positions']) > 0:
            #find the desired agent to cooperate
            closest_agent = self.closest_agent(self.beliefs['agent_position'], self.beliefs['agent_positions'])
            self.desires['cooperative_agent'] = self.beliefs['absolute_obs'][closest_agent[0]] # closest agent's observations

            closest_prey = self.closest_prey(self.beliefs['agent_position'], self.beliefs['prey_positions'])
            self.desires['closest_prey'] = closest_prey # closest prey's position
        else:
            if len(self.beliefs['absolute_obs']) > 1:
                # order the agents by distance (first agent is the closest) and choose the closest agent with at least 2 preys
                agents = []
                for agent in self.beliefs['absolute_obs']: # agent is agent_id 
                    agents.append((agent, cityblock(self.beliefs['agent_position'], self.beliefs['absolute_obs'][agent][0][1])))
                agents.sort(key=lambda x: x[1])
                for agent in agents:
                    if len(self.beliefs['absolute_obs'][agent[0]][2]) >= 2:
                        self.desires['closest_prey'] = self.beliefs['absolute_obs'][agent[0]][0][1]
                        break
                if 'closest_prey' not in self.desires:
                    for agent in agents:
                        if len(self.beliefs['absolute_obs'][agent[0]][2]) == 1:
                            self.desires['closest_prey'] = self.beliefs['absolute_obs'][agent[0]][0][1]
                            break
                if 'closest_prey' not in self.desires:
                    get_closest_agent_relative()  
            else:
                get_closest_agent_relative()

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

    def closest_agent(self, agent_position, agent_positions):
        """
        Given the positions of an agent and a sequence of [index, positions] of all agents,
        returns the closest agent.
        If there are no agents, None is returned instead
        """
        min = math.inf
        closest_agent = None
        for agent in agent_positions:
            distance = cityblock(agent_position, agent[1])
            if distance < min:
                min = distance
                closest_agent = agent
        return closest_agent

    def closest_prey(self, agent_position, prey_positions):
        """
        Given the positions of an agent and a sequence of positions of all prey,
        returns the positions of the closest prey.
        If there are no preys, None is returned instead
        """
        min = math.inf
        closest_prey_position = None
        for prey in prey_positions:
            distance = cityblock(agent_position, prey)
            if distance < min:
                min = distance
                closest_prey = prey
        return closest_prey