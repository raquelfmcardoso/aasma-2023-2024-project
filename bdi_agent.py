import numpy as np
import math
import random
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
        #print(f"\tAgent Position: {self.beliefs['agent_position']}\n")
        #print(f"\tAgents Positions: {self.beliefs['agent_positions']}\n")
        #print(f"\Prey Positions: {self.beliefs['prey_positions']}\n")
        #print(f"\Wall Positions: {self.beliefs['wall_positions']}\n")

        # Finish absolute observations to assign cooperations
        for agent_id in range(n_agents):
            self.compute_absolute_observations(agent_id)
        self.assign_cooperation()

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
                else:
                    for agent in agents:
                        if agent[0] == 1:
                            self.desires['desired_location'] = agent[2]
                            #print(f"(Relative) Desired Location =1: {self.desires['desired_location']}\n")
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
            # ver oq fazer nos beliefs qd se ta a cooperar

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