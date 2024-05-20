import numpy as np
import math
import random
from aasma import Agent
from gym import Env
from scipy.spatial.distance import cityblock

N_ACTIONS = 5
DOWN, LEFT, UP, RIGHT, STAY = range(N_ACTIONS)


class BdiAgent(Agent):
    def __init__(self, agent_id):
        # Initialize agent's beliefs, desires, and intentions
        self.agent_id = agent_id
        self.n_actions = N_ACTIONS
        self.beliefs = {}
        self.desires = {}
        self.intentions = {}
        self.cooperating = False

    def perceive(self):
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
            if len(agent_observation) != 4:
                continue
            if (agent_observation[0][0] not in agent_id_absolute_obs): # if we can't see the agent
                agent_id_relative_obs[tuple(agent_observation[0][1])] = len(self.observation[agent_observation[0][0]][2]) # {agent_position} : {amount_of_preys}

        self.beliefs['absolute_obs'] = agent_id_absolute_obs
        self.beliefs['relative_obs'] = agent_id_relative_obs
        #print(f"\tAgent Position: {self.beliefs['agent_position']}\n")
        #print(f"\tAgents Positions: {self.beliefs['agent_positions']}\n")
        #print(f"\Prey Positions: {self.beliefs['prey_positions']}\n")
        #print(f"\Wall Positions: {self.beliefs['wall_positions']}\n")

    def update_desires(self):
        # Update agent's desires based on its beliefs
        def get_closest_agent_relative():
            #print("Checking relative agents")
            if len(self.beliefs['relative_obs']) > 0:
                agents = []
                for agent_pos, num_preys in self.beliefs['relative_obs'].items():
                    agents.append((num_preys, cityblock(self.beliefs['agent_position'], agent_pos), agent_pos))
                agents.sort(key=lambda x: x[1])
                for agent in agents:
                    if agent[0] >= 2:
                        self.desires['desired_location'] = agent[2]
                        #print(f"(Relative) Desired Location >=2: {self.desires['desired_location']}\n")
                        break
                if 'desired_location' not in self.desires:
                    for agent in agents:
                        if agent[0] == 1:
                            self.desires['desired_location'] = agent[2]
                            #print(f"(Relative) Desired Location =1: {self.desires['desired_location']}\n")
                            break
                # if 'desired_location' not in self.desires then 'closest_prey' is not as well and we'll move randomly
            # if we can't see a relative agent, we'll move randomly

        if not self.cooperating:
            if len(self.beliefs['prey_positions']) > 0:
                #find the desired agent to cooperate
                #print("At least one prey was seen")
                closest_agent = self.closest_agent(self.beliefs['agent_position'], self.beliefs['agent_positions'])
                if closest_agent:
                    #print(f"BBBBBBB: id: {closest_agent[0]} and {self.beliefs['absolute_obs']} and {self.beliefs['agent_positions']}")
                    self.desires['cooperative_agent'] = self.beliefs['absolute_obs'][closest_agent[0]] # closest agent's observations
                else:
                    self.desires['cooperative_agent'] = []
                #print(f"\Cooperative Agent: {self.desires['cooperative_agent']}\n")

                closest_prey = self.closest_prey(self.beliefs['agent_position'], self.beliefs['prey_positions'])
                self.desires['closest_prey'] = closest_prey # closest prey's position
                #print(f"\Closest Prey: {self.desires['closest_prey']}\n")
            else:
                #print("No prey was seen")
                if len(self.beliefs['absolute_obs']) > 1:
                    #print("At least one close agent")
                    # order the agents by distance (first agent is the closest) and choose the closest agent with at least 2 preys
                    agents = []
                    for agent in self.beliefs['absolute_obs']: # agent is agent_id
                        #print(f"AAAAAAAAAAAAAAA: id: {agent} and {self.beliefs['absolute_obs'][agent]}")
                        agents.append((agent, cityblock(self.beliefs['agent_position'], self.beliefs['absolute_obs'][agent][0][1])))
                    agents.sort(key=lambda x: x[1]) 
                    for agent in agents:
                        if len(self.beliefs['absolute_obs'][agent[0]][2]) >= 2:
                            self.desires['desired_location'] = self.beliefs['absolute_obs'][agent[0]][0][1]
                            #print(f"\Desired Location >=2: {self.desires['desired_location']}\n")
                            break
                    if 'desired_location' not in self.desires:
                        for agent in agents:
                            if len(self.beliefs['absolute_obs'][agent[0]][2]) == 1:
                                self.desires['desired_location'] = self.beliefs['absolute_obs'][agent[0]][0][1]
                                #print(f"\Desired Location =1: {self.desires['desired_location']}\n")
                                break
                    if 'desired_location' not in self.desires:
                        get_closest_agent_relative()
                    
                else:
                    get_closest_agent_relative()
        else:
            pass
            # ver oq fazer nos beliefs qd se ta a cooperar

    def deliberation(self):    
        # Select intentions based on agent's desires and beliefs
        if not self.cooperating:
            if 'desired_location' in self.desires:
                self.intentions['location'] = self.desires['desired_location']
                #print(f"\Location: {self.intentions['location']}\n")
                # we aren't going to cooperate in this step and therefore we will only move to the desired location
            elif 'cooperative_agent' in self.desires:
                self.intentions['cooperate'] = self.desires['cooperative_agent']
                self.intentions['hunt'] = self.desires['closest_prey']
                #print(f"\Cooperate: {self.intentions['cooperate']}\n")
                #print(f"\Hunt: {self.intentions['hunt']}\n")
                # implement logic to cooperate and hunt
            elif 'desired_location' and 'closest_prey' not in self.desires:
                # we didn't find any agent to cooperate, any agent to follow or any prey to hunt
                self.intentions['location'] = 'random'
                #print(f"\Random Location: {self.intentions['location']}\n")
        else:
            pass

    def action(self):
        agents_positions = [agent[1] for agent in self.beliefs['agent_positions']]
        wall_agents_positions = agents_positions + self.beliefs['wall_positions']

        moves = {
                 DOWN: move if (move := self._apply_move([self.beliefs['agent_position'][0], self.beliefs['agent_position'][1]], DOWN)) not in wall_agents_positions else None,
                 LEFT: move if (move := self._apply_move([self.beliefs['agent_position'][0], self.beliefs['agent_position'][1]], LEFT)) not in wall_agents_positions else None,
                 UP: move if (move := self._apply_move([self.beliefs['agent_position'][0], self.beliefs['agent_position'][1]], UP)) not in wall_agents_positions else None,
                 RIGHT: move if (move := self._apply_move([self.beliefs['agent_position'][0], self.beliefs['agent_position'][1]], RIGHT)) not in wall_agents_positions else None,
                 STAY: [self.beliefs['agent_position'][0], self.beliefs['agent_position'][1]]
                }
        # Only account for moves that don't try to move into a wall
        possible_moves = [x for x in moves if moves[x] != None]
        
        if 'location' in self.intentions:
            if self.intentions['location'] == 'random': # move randomly
                move = np.random.choice(possible_moves)
                #print(f"\Move Randomly: {move}\n")
            else: # move to the desired location
                move = self.direction_to_go(self.beliefs['agent_position'], self.intentions['location'], possible_moves)
                #print(f"\Move Location: {move}\n")
        elif 'hunt' in self.intentions:
            move = self.direction_to_go(self.beliefs['agent_position'], self.intentions['hunt'], possible_moves)
            #print(f"\Move Hunt: {move}\n")
        else: # neither is there so move randomly
            move = np.random.choice(possible_moves)
            #print(f"\Move Randomly 2: {move}\n")
        return move
        
    def run(self):
        # Run the BDI agent's decision-making loop
        self.perceive()
        self.update_desires()
        self.deliberation()
        action = self.action()
        self.beliefs = {}
        self.desires = {}
        self.intentions = {}
        return action

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
    
    def direction_to_go(self, agent_position, destination_position, possible_moves):
        """
        Given the position of the agent and a destination position,
        returns the action to take in order to close the distance
        """
        distances = np.array(destination_position) - np.array(agent_position)
        # print(f"AGENT: Distance between agent {agent_position} and prey {destination_position}: {distances}")
        abs_distances = np.absolute(distances)
        if abs_distances[0] > abs_distances[1]:
            return self._close_vertically(distances, possible_moves, True)
        elif abs_distances[0] < abs_distances[1]:
            return self._close_horizontally(distances, possible_moves, True)
        else:
            roll = random.uniform(0, 1)
            return self._close_horizontally(distances, possible_moves, True) if roll > 0.5 else self._close_vertically(distances, possible_moves, True)
    
    def _apply_move(self, agent_position, move):
        if move == RIGHT:
            return [agent_position[0], agent_position[1] + 1]
        elif move == LEFT:
            return [agent_position[0], agent_position[1] - 1]
        elif move == UP:
            return [agent_position[0] - 1, agent_position[1]]
        elif move == DOWN:
            return [agent_position[0] + 1, agent_position[1]]
        else:
            return [agent_position[0], agent_position[1]]

    def _close_horizontally(self, distances, possible_moves, first_call):
        if distances[1] > 0 and RIGHT in possible_moves:
            return RIGHT
        elif distances[1] < 0 and LEFT in possible_moves:
            return LEFT
        # If we haven't yet tried the other axis
        elif first_call:
            # Remove the moves we've tried
            if LEFT in possible_moves:
                possible_moves.remove(LEFT)
            if RIGHT in possible_moves:
                possible_moves.remove(RIGHT)
            # Try to move in the other axis if possible
            return self._close_vertically(distances=distances, possible_moves=possible_moves, first_call=False)
        # This happens if we're next to a wall, but aligned with the predator, meaning we don't care in which way we go, as long as we move
        elif len(possible_moves) != 1:
            possible_moves.remove(STAY)
            return random.choice(possible_moves)
        # If we're stuck/the only moves available put us closer to the predator
        else:
            return STAY

    def _close_vertically(self, distances, possible_moves, first_call):
        if distances[0] > 0 and DOWN in possible_moves:
            return DOWN
        elif distances[0] < 0 and UP in possible_moves:
            return UP
        elif first_call:
            if DOWN in possible_moves:
                possible_moves.remove(DOWN)
            if UP in possible_moves:
                possible_moves.remove(UP)
            return self._close_horizontally(distances=distances, possible_moves=possible_moves, first_call=False)
        elif len(possible_moves) != 1:
            possible_moves.remove(STAY)
            return random.choice(possible_moves)
        else:
            return STAY