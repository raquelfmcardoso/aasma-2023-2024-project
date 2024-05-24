import numpy as np
import math
import random
from aasma import Agent
from gym import Env
from scipy.spatial.distance import cityblock

N_ACTIONS = 5
DOWN, LEFT, UP, RIGHT, STAY = range(N_ACTIONS)

def deepcopy(to_copy):
    if (type(to_copy) == list):
        return [deepcopy(element) for element in to_copy]
    else:
        return to_copy

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

    def assign_cooperations(self, agent_id):

        # {agent_id} = [[{coordenation_agent_id}, [{coordenation_agent_coords}]], [{deired_prey_coords}]]
        if agent_id in self.beliefs['cooperations']:
            if (agent_id == self.agent_id and self.beliefs['cooperations'][agent_id][0] != None):
                self.cooperating = True
            return

        agent_absolute_obs = self.beliefs['absolute_obs'][agent_id]
        agent_coords = agent_absolute_obs[0][1]

        min_prey_distance = math.inf
        prey_coordenation_agent_id = None
        prey_coordenation_agent_coords = []
        min_agent_distance = math.inf
        agent_coordenation_agent_id = None
        agent_coordenation_agent_coords = []
        desired_prey_coords = []

        for seen_agent in agent_absolute_obs[1]:
            seen_agent_id = seen_agent[0]
            if seen_agent_id == agent_id:
                continue
            # Check if the other agent already has a coordination
            if (seen_agent_id in self.beliefs['cooperations'] and self.beliefs['cooperations'][seen_agent_id][0][0] != None):
                continue
            seen_agent_coords = seen_agent[1]
            average_agent_coords = [round((agent_coords[0] + seen_agent_coords[0])/2), round((agent_coords[1] + seen_agent_coords[1])/2)]
            # Since both agents see each other, they share the same observations
            seen_preys = agent_absolute_obs[2]
            # If we see preys, pick based on the closest prey
            if len(seen_preys) > 0:
                for seen_prey in seen_preys:
                    if ((distance := cityblock(average_agent_coords, seen_prey)) < min_prey_distance):
                        min_prey_distance = distance
                        prey_coordenation_agent_id = seen_agent_id
                        prey_coordenation_agent_coords = seen_agent_coords
                        desired_prey_coords = seen_prey
            # If we don't see preys, pick based on the closest agent
            elif prey_coordenation_agent_id == None:
                if ((distance := cityblock(agent_coords, seen_agent_coords)) < min_agent_distance):
                    min_agent_distance = distance
                    agent_coordenation_agent_id = seen_agent_id
                    agent_coordenation_agent_coords = seen_agent_coords

        # If we didn't find a prey to hunt together
        if (prey_coordenation_agent_id == None):
            if agent_id == self.agent_id:
                self.cooperating = False
            self.beliefs['cooperations'][agent_id] = [[agent_coordenation_agent_id, agent_coordenation_agent_coords], None]
            # If we found at least found a free agent, we colaborate
            if (agent_coordenation_agent_id != None):
                if agent_id == self.agent_id:
                    self.cooperating = True
                self.beliefs['cooperations'][agent_coordenation_agent_id] = [[agent_id, agent_coords], None]
        # If we found a prey
        else:
            if agent_id == self.agent_id:
                self.cooperating = True
            self.beliefs['cooperations'][agent_id] = [[prey_coordenation_agent_id, prey_coordenation_agent_coords], desired_prey_coords]
            self.beliefs['cooperations'][prey_coordenation_agent_id] = [[agent_id, agent_coords], desired_prey_coords]

    def join_absolute_observations(self, agent_id, seen_agent_id):
        # Adds observations from seen_agent_id to agent_id
        for tmp_agent in self.beliefs['absolute_obs'][seen_agent_id][1]:
            if tmp_agent not in self.beliefs['absolute_obs'][agent_id][1]:
                self.beliefs['absolute_obs'][agent_id][1].append(tmp_agent)
        for tmp_prey in self.beliefs['absolute_obs'][seen_agent_id][2]:
            if tmp_prey not in self.beliefs['absolute_obs'][agent_id][2]:
                self.beliefs['absolute_obs'][agent_id][2].append(tmp_prey)
        for tmp_wall in self.beliefs['absolute_obs'][seen_agent_id][3]:
            if tmp_wall not in self.beliefs['absolute_obs'][agent_id][3]:
                self.beliefs['absolute_obs'][agent_id][3].append(tmp_wall)

    def compute_absolute_observations(self, agent_id):
        if agent_id not in self.beliefs['absolute_obs']:
            self.beliefs['absolute_obs'][agent_id] = deepcopy(self.observation[agent_id])
            self.beliefs['absolute_obs'][agent_id][1].insert(0, self.beliefs['absolute_obs'][agent_id][0])
        else:
            return

        # For each agent that agent_id can see
        for seen_agent_n in range(len(self.beliefs['absolute_obs'][agent_id][1])):
            seen_agent = self.beliefs['absolute_obs'][agent_id][1][seen_agent_n]
            seen_agent_id = seen_agent[0]
            if seen_agent_id == agent_id:
                continue
            # If that agent's absolute observations have been computed
            if seen_agent_id in self.beliefs['absolute_obs'] and len(self.beliefs['absolute_obs'][seen_agent_id]) != 0:
                # Add the ones that we don't have to ours and add ours to the other one
                self.join_absolute_observations(seen_agent_id, agent_id)
                self.join_absolute_observations(agent_id, seen_agent_id)
            else:
                self.compute_absolute_observations(seen_agent_id)
                self.join_absolute_observations(seen_agent_id, agent_id)
                self.join_absolute_observations(agent_id, seen_agent_id)

    def perceive(self):
        # Update agent's beliefs based on the current environment
        agent_position = deepcopy(self.observation[self.agent_id][0][1])
        agent_positions = deepcopy(self.observation[self.agent_id][1])
        prey_positions = deepcopy(self.observation[self.agent_id][2])
        wall_positions = deepcopy(self.observation[self.agent_id][3])
        self.beliefs['observations'] = deepcopy(self.observation)
        self.beliefs['agent_position'] = agent_position
        self.beliefs['agent_positions'] = agent_positions
        self.beliefs['prey_positions'] = prey_positions
        self.beliefs['wall_positions'] = wall_positions
        self.beliefs['absolute_obs'] = {}
        self.beliefs['relative_obs'] = {}
        self.beliefs['cooperations'] = {}

        # We only get the amount of preys an agent we can't see can see
        agent_id_relative_obs = {}

        # Compute our own absolute observations
        self.compute_absolute_observations(self.agent_id)

        n_agents = len(self.observation)
        # Compute our relative observations (agents not included in absolute observations aren't seen by us)
        for agent_id in range(n_agents):
            if len(self.observation[agent_id]) != 4:
                continue
            # If we can't see the agent
            if (agent_id not in self.beliefs['absolute_obs']):
                # {agent_id} : [{amount_of_preys}, [{agent_coordenates}]]
                agent_id_relative_obs[agent_id] = [len(self.observation[agent_id][2]), self.observation[agent_id][0][1]]

        self.beliefs['relative_obs'] = agent_id_relative_obs

        # Finish absolute observations to assign cooperations
        for agent_id in range(n_agents):
            if len(self.observation[agent_id]) != 4:
                continue
            self.compute_absolute_observations(agent_id)

        for agent_id in self.conventions:
            if len(self.observation[agent_id]) != 4:
                continue
            self.assign_cooperations(agent_id)

    def update_desires(self):
        # Update agent's desires based on its beliefs
        if not self.cooperating:
            # Find the [ids, coords] of agents that don't have a cooperation bond
            non_cooperating_agents = [agent_id for agent_id in self.beliefs['cooperations'] \
                                      if self.beliefs['cooperations'][agent_id][0][0] == None and agent_id != self.agent_id]

            # If all other agents have a cooperation, the agent doesn't desire anything and will move randomly
            if (len(non_cooperating_agents) == 0):
                return

            # Find the closest bondless agent
            distances = []
            for agent in non_cooperating_agents:
                other_agent_coords = self.beliefs['observations'][agent][0][1]
                distances.append([other_agent_coords, cityblock(self.beliefs['agent_position'], other_agent_coords)])

            distances.sort(key=lambda x: x[1])

            # Move towards the closest agent
            self.desires['desired_location'] = distances[0][0]
        # If we're collaborating and chasing a prey
        elif self.beliefs['cooperations'][self.agent_id][1] != None:
            cooperation_agent_id = self.beliefs['cooperations'][self.agent_id][0][0]
            desired_prey_coords = self.beliefs['cooperations'][self.agent_id][1]
            seen_preys = self.beliefs['absolute_obs'][self.agent_id][2]
            seen_walls = self.beliefs['absolute_obs'][self.agent_id][3]

            moves = {
                 DOWN: move if (move := self._apply_move(desired_prey_coords, DOWN)) not in seen_walls and move not in seen_preys else None,
                 LEFT: move if (move := self._apply_move(desired_prey_coords, LEFT)) not in seen_walls and move not in seen_preys else None,
                 UP: move if (move := self._apply_move(desired_prey_coords, UP)) not in seen_walls and move not in seen_preys else None,
                 RIGHT: move if (move := self._apply_move(desired_prey_coords, RIGHT)) not in seen_walls and move not in seen_preys else None
            }
            possible_moves = [x for x in moves if moves[x] != None]
            
            # There is only 1 available move, so we race to it
            if len(possible_moves) == 1:
                self.desires['desired_location'] = moves[possible_moves[0]]
                return
            
            # Can't reach prey
            if len(possible_moves) == 0:
                return
            
            our_distances = []
            other_distances = []
            our_agent_coords = self.beliefs['agent_position']
            cooperation_agent_coords = self.beliefs['cooperations'][self.agent_id][0][1]

            for move in possible_moves:
                our_distances.append([move, cityblock(moves[move], our_agent_coords)])
                other_distances.append([move, cityblock(moves[move], cooperation_agent_coords)])
          
            our_distances.sort(key=lambda x: x[1])
            other_distances.sort(key=lambda x: x[1])

            if (our_distances[0][0] == other_distances[0][0]):
                order_self = self.conventions.index(self.agent_id)
                order_other = self.conventions.index(cooperation_agent_id)

                iterator_1 = our_distances if order_self < order_other else other_distances
                iterator_2 = other_distances if order_self < order_other else our_distances

                min = math.inf
                move1 = STAY
                move2 = STAY
                for distance_pair in iterator_1:
                    for distance_pair2 in iterator_2:
                        if (distance_pair[0] == distance_pair2[0]):
                            continue
                        if ((distance_sum := distance_pair[1] + distance_pair2[1]) < min):
                            min = distance_sum
                            move1 = distance_pair[0]
                            move2 = distance_pair2[0]
                        # Since the lists are sorted, we only have to evaluate one value
                        break
                if (move1 == STAY or move2 == STAY):
                    print(f"SOMETHING WENT WRONG WHEN PICKING AGENT'S MOVES CHASING A PREY.")
                self.desires['desired_location'] = moves[move1] if order_self < order_other else moves[move2]
            else:
                self.desires['desired_location'] = moves[our_distances[0][0]]
        # We're collaborating, but don't see any preys
        else:
            cooperation_agent_id = self.beliefs['cooperations'][self.agent_id][0][0]
            seen_walls = self.beliefs['absolute_obs'][self.agent_id][3]
            relative_obs = self.beliefs['relative_obs']
            # {agent_id} : [{amount_of_preys}, [{agent_coordenates}]]
            possible_agents_1 = []
            possible_agents_2 = []
            for agent_id in relative_obs:
                # We prefer agents that see at least 2 preys so we minimize the chances
                # of it being already eaten when we get there
                if (relative_obs[agent_id][0] >= 2):
                    possible_agents_2.append(agent_id)
                elif (relative_obs[agent_id][0] == 1):
                    possible_agents_1.append(agent_id)

            our_agent_coords = self.beliefs['agent_position']
            other_agent_coords = self.beliefs['cooperations'][self.agent_id][0][1]
            # If we don't have any relative observations, then we both move in an arbitrary direction
            if len(possible_agents_2) == 0 and len(possible_agents_1) == 0:
                moves = {
                 DOWN: move if (move := self._apply_move(our_agent_coords, DOWN)) not in seen_walls else None,
                 LEFT: move if (move := self._apply_move(our_agent_coords, LEFT)) not in seen_walls else None,
                 UP: move if (move := self._apply_move(our_agent_coords, UP)) not in seen_walls else None,
                 RIGHT: move if (move := self._apply_move(our_agent_coords, RIGHT)) not in seen_walls else None
                }
                other_moves = {
                 DOWN: move if (move := self._apply_move(other_agent_coords, DOWN)) not in seen_walls else None,
                 LEFT: move if (move := self._apply_move(other_agent_coords, LEFT)) not in seen_walls else None,
                 UP: move if (move := self._apply_move(other_agent_coords, UP)) not in seen_walls else None,
                 RIGHT: move if (move := self._apply_move(other_agent_coords, RIGHT)) not in seen_walls else None
                }
                possible_moves = [x for x in moves if moves[x] != None and other_moves[x] != None]

                if (size := len(possible_moves)) == 0:
                    print(f"PREDATORS ARE STUCK WHEN TRYING TO MOVE")
                    return                    

                deterministic_index = ((our_agent_coords[0] * other_agent_coords[1]) + (our_agent_coords[1] * other_agent_coords[0])) % size
                self.desires['desired_location'] = moves[possible_moves[deterministic_index]]
            # Move towards the closest agent with observations, preferably at least 2 preys
            else:
                average_coords = [round((our_agent_coords[0] + other_agent_coords[0])/2), round((our_agent_coords[1] + other_agent_coords[1])/2)]

                iterator = possible_agents_2 if len(possible_agents_2) > 0 else possible_agents_1
                distances = []
                for agent_id in iterator:
                    distances.append([agent_id, cityblock(relative_obs[agent_id][1], average_coords)])
                distances.sort(key=lambda x: x[1])

                self.desires['desired_location'] = relative_obs[distances[0][0]][1]

    def deliberation(self):
        # Select intentions based on agent's desires and beliefs
        if 'desired_location' in self.desires:
            # Move towards the desired location
            self.intentions['location'] = self.desires['desired_location']
        else:
            # Move randomly if no location is desired
            self.intentions['location'] = 'random'

    def action(self):
        agent_positions = [x[1] for x in self.beliefs['agent_positions'] if x[0] > self.agent_id]
        prey_positions = self.beliefs['prey_positions']
        wall_positions = self.beliefs['wall_positions']

        moves = {
                 DOWN: move if (move := self._apply_move(self.beliefs['agent_position'], DOWN)) \
                    not in wall_positions and move not in prey_positions and move not in agent_positions else None,
                 LEFT: move if (move := self._apply_move(self.beliefs['agent_position'], LEFT)) \
                    not in wall_positions and move not in prey_positions and move not in agent_positions else None,
                 UP: move if (move := self._apply_move(self.beliefs['agent_position'], UP)) \
                    not in wall_positions and move not in prey_positions and move not in agent_positions else None,
                 RIGHT: move if (move := self._apply_move(self.beliefs['agent_position'], RIGHT)) \
                    not in wall_positions and move not in prey_positions and move not in agent_positions else None,
                 STAY: self.beliefs['agent_position']
                }
        # Only account for moves that don't try to move into a wall
        possible_moves = [x for x in moves if moves[x] != None]
        
        if 'location' in self.intentions:
            if self.intentions['location'] == 'random':
                move = np.random.choice(possible_moves)
            else:
                move = self.direction_to_go(self.beliefs['agent_position'], self.intentions['location'], possible_moves)
        else:
            print("NO LOCATION INTENTION DEFINED. ERROR.")
            move = np.random.choice(possible_moves)
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
        self.cooperating = False
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
        if (agent_position == destination_position):
            return STAY
        distances = np.array(destination_position) - np.array(agent_position)
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