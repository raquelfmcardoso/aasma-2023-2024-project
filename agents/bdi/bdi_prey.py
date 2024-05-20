import numpy as np
import math
import random
from aasma import Agent
from gym import Env
from scipy.spatial.distance import cityblock

N_ACTIONS = 5
DOWN, LEFT, UP, RIGHT, STAY = range(N_ACTIONS)

class BdiPrey(Agent):
    def __init__(self, prey_id, family_id):
        # Initialize prey's beliefs, desires, and intentions
        self.prey_id = prey_id
        self.n_actions = N_ACTIONS
        self.family_id = family_id # 1 -> +3 vision range & 1 speed || 2 -> normal vision range & 2 speed
        self.beliefs = {}
        self.desires = {}
        self.intentions = {}

    def perceive(self):
        # Update prey's beliefs based on the current environment
        prey_position = self.observation[self.prey_id][0][1] # [prey_x, prey_y] and its 1 item
        agent_positions = self.observation[self.prey_id][1] # each entry is [agent_x, agent_y]
        prey_positions = self.observation[self.prey_id][2] # each entry is [prey_id, [prey_x, prey_y]]
        wall_positions = self.observation[self.prey_id][3] # each entry is [wall_x, wall_y]
        self.beliefs['prey_position'] = prey_position
        self.beliefs['agent_positions'] = agent_positions
        self.beliefs['prey_positions'] = prey_positions
        self.beliefs['wall_positions'] = wall_positions

        # We get the complete observations of agents we can see
        prey_id_absolute_obs = {}
        # We only get the amount of preys an agent we can't see can see
        prey_id_relative_obs = {}

        prey_id_absolute_obs[self.prey_id] = self.observation[self.prey_id] # add our own observations
        for prey in prey_positions:
            prey_id_absolute_obs[prey[0]] = self.observation[prey[0]] # {prey_id} : {self.observations[prey_id]}

        for prey_observation in self.observation:
            #check if prey_observation is a empty list
            if len(prey_observation) != 4:
                continue
            if (prey_observation[0][0] not in prey_id_absolute_obs): # if we can't see the prey
                prey_id_relative_obs[tuple(prey_observation[0][1])] = len(self.observation[prey_observation[0][0]][2]) # {prey_position} : {amount_of_agents}

        self.beliefs['absolute_obs'] = prey_id_absolute_obs
        self.beliefs['relative_obs'] = prey_id_relative_obs

        # distances to visible entities
        self.beliefs['distances_nearby_agents'] = [cityblock(prey_position, agent) for agent in agent_positions]
        self.beliefs['distances_nearby_preys'] = [cityblock(prey_position, prey[1]) for prey in prey_positions]
        self.beliefs['distances_nearby_walls'] = [cityblock(prey_position, wall) for wall in wall_positions]

        # positions of agents seen by nearby preys
        self.beliefs['unseen_agent_positions'] = []
        for prey in self.beliefs['absolute_obs']:
            for agent in self.beliefs['absolute_obs'][prey][1]: # prey is prey_id
                self.beliefs['unseen_agent_positions'].append(agent)

        #remove duplicate agent positions
        self.beliefs['unseen_agent_positions'] = [pos for pos in self.beliefs['unseen_agent_positions'] if pos not in self.beliefs['agent_positions']]
        tuple_pos = [tuple(pos) for pos in self.beliefs['unseen_agent_positions']]
        tuple_pos = set(tuple_pos)
        self.beliefs['unseen_agent_positions'] = [list(pos) for pos in tuple_pos]
        
        # distance to agents seen by nearby preys
        self.beliefs['distances_unseen_agents'] = [cityblock(prey_position, agent) for agent in self.beliefs['unseen_agent_positions']]
        
        # positions of unseen preys that see agents
        self.beliefs['unseen_prey_positions'] = []
        for prey in self.beliefs['relative_obs']: # prey is prey_position
            if self.beliefs['relative_obs'][prey] > 0:
                self.beliefs['unseen_prey_positions'].append(prey) # prey is [prey_x, prey_y]
        # distance to unseen preys
        self.beliefs['distances_unseen_preys'] = [cityblock(prey_position, prey) for prey in self.beliefs['unseen_prey_positions']]

        #print(f"\tPrey Position: {self.beliefs['prey_position']}\n")
        #print(f"\tPreys Positions: {self.beliefs['agent_positions']}\n")
        #print(f"\Agents Positions: {self.beliefs['prey_positions']}\n")
        #print(f"\Wall Positions: {self.beliefs['wall_positions']}\n")

    def update_desires(self):
        agents_positions = self.beliefs['agent_positions'] + self.beliefs['unseen_agent_positions']
        distances_agents = self.beliefs['distances_nearby_agents'] + self.beliefs['distances_unseen_agents']
        unseen_preys_positions = self.beliefs['unseen_prey_positions']
        unseen_preys_distances = self.beliefs['distances_unseen_preys']

        print(f"\tAgents Positions: {agents_positions}\n")
        print(f"\tDistances Agents: {distances_agents}\n")

        for distance in distances_agents:
            if distance == 0:
                self.desires['run_location'] = self.beliefs['prey_position'] # if an agent is on top of the prey, prey is dead
                return
        if agents_positions and distances_agents:
            weighted_x = sum(agent[0] / distance for agent, distance in zip(agents_positions, distances_agents))
            weighted_y = sum(agent[1] / distance for agent, distance in zip(agents_positions, distances_agents))
            total_weight = sum(1 / distance for distance in distances_agents)
            if total_weight != 0:
                position = [int(weighted_x / total_weight), int(weighted_y / total_weight)]
                self.desires['run_location'] = position
                print(f"\tRun Location: {position}\n")
    
        elif unseen_preys_positions and unseen_preys_distances:
            weighted_x = sum(prey[0] / distance for prey, distance in zip(unseen_preys_positions, unseen_preys_distances))
            weighted_y = sum(prey[1] / distance for prey, distance in zip(unseen_preys_positions, unseen_preys_distances))
            total_weight = sum(1 / distance for distance in unseen_preys_distances)
            if total_weight != 0:
                position = [int(weighted_x / total_weight), int(weighted_y / total_weight)]
                self.desires['run_location'] = position
                print(f"\tRelative Run Location: {position}\n")
            
    def deliberation(self):    
        # Select intentions based on agent's desires and beliefs
        if 'run_location' in self.desires:
            self.intentions['run'] = self.desires['run_location']
            #print(f"\Run: {self.intentions['run']}\n")
        else:
            self.intentions['run'] = 'random'

    def action(self):
        preys_positions = [prey[1] for prey in self.beliefs['prey_positions']]
        wall_agents_positions = preys_positions + self.beliefs['wall_positions']

        moves = {
                 DOWN: move if (move := self._apply_move([self.beliefs['prey_position'][0], self.beliefs['prey_position'][1]], DOWN)) not in wall_agents_positions else None,
                 LEFT: move if (move := self._apply_move([self.beliefs['prey_position'][0], self.beliefs['prey_position'][1]], LEFT)) not in wall_agents_positions else None,
                 UP: move if (move := self._apply_move([self.beliefs['prey_position'][0], self.beliefs['prey_position'][1]], UP)) not in wall_agents_positions else None,
                 RIGHT: move if (move := self._apply_move([self.beliefs['prey_position'][0], self.beliefs['prey_position'][1]], RIGHT)) not in wall_agents_positions else None,
                 STAY: [self.beliefs['prey_position'][0], self.beliefs['prey_position'][1]]
                }
        # Only account for moves that don't try to move into a wall
        possible_moves = [x for x in moves if moves[x] != None]
        
        if self.intentions['run'] == 'random': # move randomly
            move = np.random.choice(possible_moves)
            #print(f"\Move Randomly: {move}\n")
        else: # move to the desired location
            move = self.direction_to_go(self.beliefs['prey_position'], self.intentions['run'], possible_moves)
            #print(f"\Move Location: {move}\n")
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
    
    def direction_to_go(self, prey_position, agent_position, possible_moves):
        """
        Given the position of the agent and the position of a prey,
        returns the action to take in order to close the distance
        """
        distances = np.array(prey_position) - np.array(agent_position)
        # print(f"PREY: Distance between agent {agent_position} and prey {prey_position}: {distances}")
        abs_distances = np.absolute(distances)
        # print(f"PREY: Distance between prey @ {prey_position} and predator @ {agent_position} is {distances}")
        # TODO: Test swapping closing call, remember to call the other function in the else branch of the _close_x function
        if abs_distances[0] > abs_distances[1]:
            return self._close_vertically(distances, possible_moves, True)
        elif abs_distances[0] < abs_distances[1]:
            return self._close_horizontally(distances, possible_moves, True)
        else:
            roll = random.uniform(0, 1)
            return self._close_horizontally(distances, possible_moves, True) if roll > 0.5 else self._close_vertically(distances, possible_moves, True)
    
    def _apply_move(self, prey_position, move):        
        if move == RIGHT:
            if self.family_id == 1:
                return [prey_position[0], prey_position[1] + 1]
            if self.family_id == 2:
                if [prey_position[0], prey_position[1] + 2] in self.beliefs['wall_positions']:
                    return [prey_position[0], prey_position[1] + 1]
                if [prey_position[0], prey_position[1] + 1] in self.beliefs['wall_positions']:
                    return [prey_position[0], prey_position[1] + 1]
                return [prey_position[0], prey_position[1] + 2]
            
        elif move == LEFT:
            if self.family_id == 1:
                return [prey_position[0], prey_position[1] - 1]
            if self.family_id == 2:
                if [prey_position[0], prey_position[1] - 2] in self.beliefs['wall_positions']:
                    return [prey_position[0], prey_position[1] - 1]
                if [prey_position[0], prey_position[1] - 1] in self.beliefs['wall_positions']:
                    return [prey_position[0], prey_position[1] - 1]
                return [prey_position[0], prey_position[1] - 2]
        
        
        elif move == UP:
            if self.family_id == 1:
                return [prey_position[0] - 1, prey_position[1]]
            if self.family_id == 2:
                if [prey_position[0] - 2, prey_position[1]] in self.beliefs['wall_positions']:
                    return [prey_position[0] - 1, prey_position[1]]
                if [prey_position[0] - 1, prey_position[1]] in self.beliefs['wall_positions']:
                    return [prey_position[0] - 1, prey_position[1]]
                return [prey_position[0] - 2, prey_position[1]]
            
        elif move == DOWN:
            if self.family_id == 1:
                return [prey_position[0] + 1, prey_position[1]]
            if self.family_id == 2:
                if [prey_position[0] + 2, prey_position[1]] in self.beliefs['wall_positions']:
                    return [prey_position[0] + 1, prey_position[1]]
                if [prey_position[0] + 1, prey_position[1]] in self.beliefs['wall_positions']:
                    return [prey_position[0] + 1, prey_position[1]]
                return [prey_position[0] + 2, prey_position[1]]
            
        else:
            return [prey_position[0], prey_position[1]]

    def _close_horizontally(self, distances, possible_moves, first_call):
        if distances[1] > 0 and RIGHT in possible_moves:
            return RIGHT
        elif distances[1] < 0 and LEFT in possible_moves:
            return LEFT
        elif first_call:
            if LEFT in possible_moves:
                possible_moves.remove(LEFT)
            if RIGHT in possible_moves:
                possible_moves.remove(RIGHT)
            return self._close_vertically(distances=distances, possible_moves=possible_moves, first_call=False)
        elif len(possible_moves) != 1:
            possible_moves.remove(STAY)
            return random.choice(possible_moves)
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