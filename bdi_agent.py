import numpy as np
import math
from aasma import Agent
from gym import Env
from scipy.spatial.distance import cityblock

N_ACTIONS = 5
DOWN, LEFT, UP, RIGHT, STAY = range(N_ACTIONS)

def deepcopy(observations):
    new_obs = []
    for agent in observations:
        new_agent = [agent[0]]
        new_coords = []
        for coord in agent[1]:
            new_coords.append(coord)
        new_agent.append(new_coords)
        new_obs.append(new_agent)
    return new_obs

class BdiAgent(Agent):
    def __init__(self, agent_id, conventions):
        # Initialize agent's beliefs, desires, and intentions
        self.agent_id = agent_id
        self.n_actions = N_ACTIONS
        self.beliefs = {}
        self.desires = {}
        self.intentions = {}
        self.conventions = conventions
        self.cooperating = False

    def compute_absolute_observations(self, agent_id):
        if agent_id not in self.beliefs['absolute_obs']:
            self.beliefs['absolute_obs'][agent_id] = self.observation[agent_id]

        # For each agent that agent_id can see
        for seen_agent in self.observation[agent_id][1]:
            seen_agent_id = seen_agent[0][0]
            # If that agent's absolute observations have been computed
            if seen_agent_id in self.beliefs['absolute_obs'] and len(self.beliefs['absolute_obs'][seen_agent_id]) != 0:
                # Add the ones that we don't have to ours
                for tmp_agent in self.beliefs['absolute_obs'][seen_agent_id][1]:
                    if tmp_agent not in self.beliefs['absolute_obs'][agent_id][1]:
                        self.beliefs['absolute_obs'][agent_id][1].append(tmp_agent)
                for tmp_prey in self.beliefs['absolute_obs'][seen_agent_id][2]:
                    if tmp_prey not in self.beliefs['absolute_obs'][agent_id][2]:
                        self.beliefs['absolute_obs'][agent_id][2].append(tmp_prey)
                for tmp_wall in self.beliefs['absolute_obs'][seen_agent_id][3]:
                    if tmp_wall not in self.beliefs['absolute_obs'][agent_id][3]:
                        self.beliefs['absolute_obs'][agent_id][2].append(tmp_wall)
            else:
                self.compute_absolute_observations(seen_agent_id)

    def assign_cooperation(self):

        # Iterate over each agent
        # For each agent:
        #   min = inf
        #   coord_agent = -1
        #   For each seen agent:
        #       If the agent already has a coordination -> skip
        #       Calculate the average position
        #       For each seen prey:
        #           Calculate the distance between the prey and the average position
        #           If it's less than the minimum, assign the distance as the new minimum, the seen agent as coord_agent (and the prey as the target?)
        #   Establish cooperation between agents in self.cooperations

        for agent_id in self.conventions:
            agent_absolute_obs = self.beliefs['absolute_obs'][agent_id]
            agent_coords = agent_absolute_obs[0][1]
            min_distance = math.inf
            coordenation_agent = -1
            for seen_agent in agent_absolute_obs[1]:
                seen_agent_id = seen_agent[0][0]
                if (seen_agent_id in self.beliefs['cooperations'] and self.beliefs['cooperations'][seen_agent_id] != -1):
                    # If the agent already is cooperating with another agent
                    continue
                seen_agent_coords = seen_agent[0][1]
                average_agent_coords = [round((agent_coords[0] + seen_agent_coords[0])/2), round((agent_coords[1] + seen_agent_coords[1])/2)]
                # Since both agents see each other, they share the same observations
                seen_preys = agent_absolute_obs[2]
                for seen_prey in seen_preys:
                    seen_prey_coords = seen_prey[1]
                    distance = cityblock(average_agent_coords, seen_prey_coords)
                    if (distance < min_distance):
                        min_distance = distance
                        coordenation_agent = seen_agent_id
            self.beliefs['cooperations'][agent_id] = coordenation_agent
            if (coordenation_agent != -1):
                self.cooperating = True
                self.beliefs['cooperations'][coordenation_agent] = agent_id

    def perceive(self, environment):
        # Update agent's beliefs based on the current environment
        self.beliefs['last_seen'] = self.beliefs.get('prey_positions')

        agent_position = self.observation[self.agent_id][0][1]
        agent_positions = self.observation[self.agent_id][1]
        prey_positions = self.observation[self.agent_id][2]
        wall_positions = self.observation[self.agent_id][3]
        self.beliefs['agent_position'] = agent_position
        self.beliefs['agent_positions'] = agent_positions
        self.beliefs['prey_positions'] = prey_positions
        self.beliefs['wall_positions'] = wall_positions

        # We only get the amount of preys an agent we can't see can see
        agent_id_relative_obs = {}

        # Compute our own absolute observations
        self.compute_absolute_observations(self.agent_id)

        n_agents = len(self.observation)
        # Compute our relative observations (agents not included in absolute observations aren't seen by us)
        for agent_id in range(n_agents):
            if (agent_id not in self.beliefs['absolute_obs']): # if we can't see the agent
                # {agent_id} : [{amount_of_preys}, [{agent_coordenates}]]
                agent_id_relative_obs[agent_id] = [len(self.observation[agent_id][2]), self.observation[agent_id][0][1]]

        self.beliefs['relative_obs'] = agent_id_relative_obs

        # Finish absolute observations to assign cooperations
        for agent_id in range(n_agents):
            self.compute_absolute_observations(agent_id)
        self.assign_cooperation()

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
                        self.desires['desired_location'] = agent[2]
                        break
                else:
                    for agent in agents:
                        if agent[0] == 1:
                            self.desires['desired_location'] = agent[2]
                            break
                    else:
                        #Move randomly
                        pass
            else: 
                #Move randomly
                pass

        if not self.cooperating:
            # Find the ids of agents that don't have a cooperation bond
            non_cooperating_agents = [agent_id for agent_id in self.beliefs['cooperations'] if self.beliefs['cooperations'][agent_id] == -1 and agent_id != self.agent_id]

            # If all other agents have a cooperation
            if (len(non_cooperating_agents) == 0):
                self.random_move()

            # Find the closest
            distances = []
            for agent_id in non_cooperating_agents:
                other_agent_coords = self.observation[agent_id][0][1]
                distances.append((other_agent_coords, cityblock(self.beliefs['agent_position'], other_agent_coords)))

            distances.sort(key=lambda x: x[1])

            self.desires['desired_location'] = distances[0][0]
        else:
            pass
            # ver oq fazer nos beliefs qd se tá a cooperar

    def random_move(self):
        moves = {
                 DOWN: move if (move := self._apply_move([self.beliefs['agent_position'][0], \
                                                          self.beliefs['agent_position'][1]], DOWN)) not in self.beliefs['wall_positions'] else None,
                 LEFT: move if (move := self._apply_move([self.beliefs['agent_position'][0], \
                                                          self.beliefs['agent_position'][1]], LEFT)) not in self.beliefs['wall_positions'] else None,
                 UP: move if (move := self._apply_move([self.beliefs['agent_position'][0], \
                                                        self.beliefs['agent_position'][1]], UP)) not in self.beliefs['wall_positions'] else None,
                 RIGHT: move if (move := self._apply_move([self.beliefs['agent_position'][0], \
                                                           self.beliefs['agent_position'][1]], RIGHT)) not in self.beliefs['wall_positions'] else None,
                 STAY: [self.beliefs['agent_position'][0], self.beliefs['agent_position'][1]]
                }

    def deliberation(self):
        # Select intentions based on agent's desires and beliefs
        if not self.cooperating:
            if 'desired_location' in self.desires:
                self.intentions['move'] = self.desires['desired_location']
                # vou fazer a action nisto
            elif 'cooperative_agent' in self.desires:
                self.intentions['cooperate'] = self.desires['cooperative_agent']
            elif 'closest_prey' in self.desires:
                self.intentions['hunt'] = self.desires['closest_prey']
        else:
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
                closest_prey_position = prey
        return closest_prey_position