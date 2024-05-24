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

# def deepcopy(observations):
#     new_obs = []
#     for agent in observations:
#         new_agent = [agent[0]]
#         new_coords = []
#         for coord in agent[1]:
#             new_coords.append(coord)
#         new_agent.append(new_coords)
#         new_obs.append(new_agent)
#     return new_obs

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

    def assign_cooperations_wrong(self, agent_id):

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

        # {agent_id} = [[{coordenation_agent_id}, [{coordenation_agent_coords}]], [{deired_prey_coords}]]
        # if agent_id in self.beliefs['cooperations']:
        #     print(f"Agent {agent_id} is already cooperating with {self.beliefs['cooperations'][agent_id]}. Skipping call")
        #     if (agent_id == self.agent_id and self.beliefs['cooperations'][agent_id][0] != None):
        #         self.cooperating = True
        #     return

        agent_absolute_obs = self.beliefs['absolute_obs'][agent_id]
        agent_coords = agent_absolute_obs[0][1]

        min_prey_distance = math.inf
        prey_coordenation_agent_id = None
        prey_coordenation_agent_coords = []
        min_agent_distance = math.inf
        agent_coordenation_agent_id = None
        agent_coordenation_agent_coords = []
        desired_prey_coords = []

        agent_cooperator_id = None
        agent_cooperator_coords = []
        agent_prey_coords = None
        agent_cooperation_prey_distance = math.inf
        agent_cooperation_agent_distance = math.inf

        # Check if we have a cooperation with someone that is also cooperating with us
        if (agent_id in self.beliefs['cooperations'] and self.beliefs['cooperations'][agent_id][0][0] != None and\
            self.beliefs['cooperations'][self.beliefs['cooperations'][agent_id][0][0]][0] == agent_id):
            print(f"We, agent {agent_id} are already cooperating with agent {self.beliefs['cooperations'][agent_id][0][0]}")
            agent_cooperator_id = self.beliefs['cooperations'][agent_id][0][0]
            agent_cooperator_coords = self.beliefs['cooperations'][agent_id][0][1]
            agent_prey_coords = self.beliefs['cooperations'][agent_id][1]

            # If we don't see any prey
            if agent_prey_coords == []:
                agent_cooperation_agent_distance = cityblock(agent_coords, agent_cooperator_coords)
                print(f"We don't see prey and our distance is: {seen_agent_cooperation_agent_distance}")
            else:
                average_agent_cooperation_coords = \
                    round((agent_coords[0] + agent_cooperator_coords[0])/2), round((agent_coords[1] + agent_cooperator_coords[1])/2)
                agent_cooperation_prey_distance = cityblock(average_agent_cooperation_coords, agent_prey_coords)
                print(f"We see prey and the distance between us and the prey is: {seen_agent_cooperation_agent_distance}")
                # Since we see prey, remove all cooperations that don't see prey
                agent_cooperation_agent_distance = -1

        for seen_agent in agent_absolute_obs[1]:
            seen_agent_id = seen_agent[0]
            seen_agent_coords = seen_agent[1]
            print(f"Agent {agent_id} evaluating cooperation with agent {seen_agent_id}")
            if seen_agent_id == agent_id or seen_agent_id == agent_cooperator_id:
                print(f"Agent is itself or their coordination")
                continue
            seen_agent_cooperator_id = None
            seen_agent_cooperator_coords = []
            seen_agent_prey_coords = []
            seen_agent_cooperation_agent_distance = math.inf
            seen_agent_cooperation_prey_distance = math.inf
            # Check if the other agent already has a coordination
            if (seen_agent_id in self.beliefs['cooperations'] and self.beliefs['cooperations'][seen_agent_id][0][0] != None):
                print(f"Agent {seen_agent_id} is already cooperating with agent {self.beliefs['cooperations'][seen_agent_id][0][0]}")
                if self.beliefs['cooperations'][seen_agent_id][0][0] == agent_id:
                    continue
                seen_agent_cooperator_id = self.beliefs['cooperations'][seen_agent_id][0][0]
                seen_agent_cooperator_coords = self.beliefs['cooperations'][seen_agent_id][0][1]
                seen_agent_prey = self.beliefs['cooperations'][seen_agent_id][1]

                if seen_agent_prey_coords == []:
                    seen_agent_cooperation_agent_distance = cityblock(seen_agent_coords, seen_agent_cooperator_coords)
                    print(f"They don't see prey and the distance for those agents is: {seen_agent_cooperation_agent_distance}")
                else:
                    average_seen_agent_cooperation_coords = \
                        round((seen_agent_coords[0] + seen_agent_cooperator_coords[0])/2), round((seen_agent_coords[1] + seen_agent_cooperator_coords[1])/2)
                    seen_agent_cooperation_prey_distance = cityblock(average_seen_agent_cooperation_coords, seen_agent_prey)
                    seen_agent_cooperation_agent_distance = -1
                    print(f"They see prey and the distance for those agents and their prey is: {seen_agent_cooperation_prey_distance}")
            average_agent_coords = [round((agent_coords[0] + seen_agent_coords[0])/2), round((agent_coords[1] + seen_agent_coords[1])/2)]
            # Since both agents see each other, they share the same observations
            seen_preys = agent_absolute_obs[2]
            # If we see preys, pick based on the closest prey
            if len(seen_preys) > 0:
                print(f"{len(seen_preys)} preys were found between agent {seen_agent_id} and agent {agent_id}'s observations")
                for seen_prey in seen_preys:
                    print(f"average_agent_coords = {average_agent_coords}; seen_prey = {seen_prey}")
                    if ((distance := cityblock(average_agent_coords, seen_prey)) < min_prey_distance\
                         and distance < seen_agent_cooperation_prey_distance and distance < agent_cooperation_prey_distance):
                        print(f"New min distance={distance} found between pair of agents {[agent_id, seen_agent_id]} and prey at {seen_prey}")
                        min_prey_distance = distance
                        prey_coordenation_agent_id = seen_agent_id
                        prey_coordenation_agent_coords = seen_agent_coords
                        desired_prey_coords = seen_prey
            # If we don't see preys, pick based on the closest agent
            elif prey_coordenation_agent_id == None:
                print(f"Agent {agent_id} and {seen_agent_id} didn't find any preys")
                if ((distance := cityblock(agent_coords, seen_agent_coords)) < min_agent_distance\
                     and distance < seen_agent_cooperation_agent_distance and distance < agent_cooperation_agent_distance):
                    print(f"New min distance={distance} found between pair of agents {[agent_id, seen_agent_id]}")
                    min_agent_distance = distance
                    agent_coordenation_agent_id = seen_agent_id
                    agent_coordenation_agent_coords = seen_agent_coords

        if (prey_coordenation_agent_id != None):
            was_changed = prey_coordenation_agent_id in self.beliefs['cooperations'] and self.beliefs['cooperations'][prey_coordenation_agent_id][0][0] != None
            self.cooperating = True
            self.beliefs['cooperations'][agent_id] = [[prey_coordenation_agent_id, prey_coordenation_agent_coords], desired_prey_coords]
            self.beliefs['cooperations'][prey_coordenation_agent_id] = [[agent_id, agent_coords], desired_prey_coords]

            if (was_changed):
                print(f"Agent {agent_id} found a better prey collaboration with {prey_coordenation_agent_id} who was already collaborating")
                self.assign_cooperations_wrong(prey_coordenation_agent_id)
            if (agent_cooperator_id != None):
                print(f"Agent {agent_id} was already collaborating with {agent_cooperator_id} but found a better collaboration with {prey_coordenation_agent_id}")
                self.assign_cooperations_wrong(agent_cooperator_id)
        elif (agent_coordenation_agent_id != None):
            was_changed = agent_coordenation_agent_id in self.beliefs['cooperations'] and self.beliefs['cooperations'][agent_coordenation_agent_id][0][0] != None
            self.cooperating = True
            self.beliefs['cooperations'][agent_id] = [[agent_coordenation_agent_id, agent_coordenation_agent_coords], None]
            self.beliefs['cooperations'][agent_coordenation_agent_id] = [[agent_id, agent_coords], None]

            if (was_changed):
                print(f"Agent {agent_id} found a better agent collaboration with {agent_coordenation_agent_id} who was already collaborating")
                self.assign_cooperations_wrong(prey_coordenation_agent_id)
            if (agent_cooperator_id != None):
                print(f"Agent {agent_id} was already collaborating with {agent_cooperator_id} but found a better collaboration with {agent_coordenation_agent_id}")
                self.assign_cooperations_wrong(agent_cooperator_id)
        elif agent_cooperator_id == None:
            self.cooperating = False
        else:
            print(f"Entered else statement in collaboration with: \
                  self.cooperation={self.cooperating}, \
                    agent_cooperator_id={agent_cooperator_id}, \
                        agent_coordenation_agent_id={agent_coordenation_agent_id}, \
                            prey_coordenation_agent_id={prey_coordenation_agent_id}")

        print(f"Finished cooperations for agent {agent_id}, cooperations: {self.beliefs['cooperations']}")

    def assign_cooperations(self, agent_id):

        # {agent_id} = [[{coordenation_agent_id}, [{coordenation_agent_coords}]], [{deired_prey_coords}]]
        if agent_id in self.beliefs['cooperations']:
            print(f"Agent {agent_id} is already cooperating with {self.beliefs['cooperations'][agent_id]}. Skipping call")
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
            print(f"Agent {agent_id} evaluating cooperation with agent {seen_agent_id}")
            if seen_agent_id == agent_id:
                print(f"Agent is itself")
                continue
            print(f"Seen agent {seen_agent_id}'s cooperation: {None if seen_agent_id not in self.beliefs['cooperations'] else self.beliefs['cooperations'][seen_agent_id]}")
            # Check if the other agent already has a coordination
            if (seen_agent_id in self.beliefs['cooperations'] and self.beliefs['cooperations'][seen_agent_id][0][0] != None):
                print(f"Agent {seen_agent_id} is already cooperating with agent {self.beliefs['cooperations'][seen_agent_id][0][0]}")
                continue
            seen_agent_coords = seen_agent[1]
            average_agent_coords = [round((agent_coords[0] + seen_agent_coords[0])/2), round((agent_coords[1] + seen_agent_coords[1])/2)]
            # Since both agents see each other, they share the same observations
            seen_preys = agent_absolute_obs[2]
            # If we see preys, pick based on the closest prey
            if len(seen_preys) > 0:
                print(f"{len(seen_preys)} preys were found between agent {seen_agent_id} and agent {agent_id}'s observations")
                for seen_prey in seen_preys:
                    print(f"average_agent_coords = {average_agent_coords}; seen_prey = {seen_prey}")
                    if ((distance := cityblock(average_agent_coords, seen_prey)) < min_prey_distance):
                        print(f"New min distance={distance} found between pair of agents {[agent_id, seen_agent_id]} and prey at {seen_prey}")
                        min_prey_distance = distance
                        prey_coordenation_agent_id = seen_agent_id
                        prey_coordenation_agent_coords = seen_agent_coords
                        desired_prey_coords = seen_prey
            # If we don't see preys, pick based on the closest agent
            elif prey_coordenation_agent_id == None:
                print(f"Agent {agent_id} and {seen_agent_id} didn't find any preys")
                if ((distance := cityblock(agent_coords, seen_agent_coords)) < min_agent_distance):
                    print(f"New min distance={distance} found between pair of agents {[agent_id, seen_agent_id]}")
                    min_agent_distance = distance
                    agent_coordenation_agent_id = seen_agent_id
                    agent_coordenation_agent_coords = seen_agent_coords

        # If we didn't find a prey to hunt together
        if (prey_coordenation_agent_id == None):
            print(f"Agent {agent_id} didn't find any hunters to hunt prey with")
            if agent_id == self.agent_id:
                self.cooperating = False
            self.beliefs['cooperations'][agent_id] = [[agent_coordenation_agent_id, agent_coordenation_agent_coords], None]
            # If we found at least found a free agent, we colaborate
            if (agent_coordenation_agent_id != None):
                print(f"But found {agent_coordenation_agent_id} to cooperate with")
                if agent_id == self.agent_id:
                    self.cooperating = True
                self.beliefs['cooperations'][agent_coordenation_agent_id] = [[agent_id, agent_coords], None]
                print(f"Updated itself to {self.beliefs['cooperations'][agent_id]} and other to {self.beliefs['cooperations'][agent_coordenation_agent_id]}")
        # If we found a prey
        else:
            print(f"Agent {agent_id} found hunter {prey_coordenation_agent_id} to hunt prey at {desired_prey_coords}")
            if agent_id == self.agent_id:
                self.cooperating = True
            self.beliefs['cooperations'][agent_id] = [[prey_coordenation_agent_id, prey_coordenation_agent_coords], desired_prey_coords]
            self.beliefs['cooperations'][prey_coordenation_agent_id] = [[agent_id, agent_coords], desired_prey_coords]
            print(f"Updated itself to {self.beliefs['cooperations'][agent_id]} and other to {self.beliefs['cooperations'][prey_coordenation_agent_id]}")

        print(f"Finished cooperations for agent {agent_id}, cooperations: {self.beliefs['cooperations']}")

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
        #print(f"Computing absolute observations for agent {agent_id}")
        if agent_id not in self.beliefs['absolute_obs']:
            self.beliefs['absolute_obs'][agent_id] = deepcopy(self.observation[agent_id])
            self.beliefs['absolute_obs'][agent_id][1].insert(0, self.beliefs['absolute_obs'][agent_id][0])
            #print(f"Agent {agent_id} added its own observations: {self.beliefs['absolute_obs']}")
        else:
            #print(f"Agent {agent_id} already had observations computed before")
            return

        #print(f"Iterating over {self.beliefs['absolute_obs'][agent_id][1]} for {agent_id}")
        # For each agent that agent_id can see
        for seen_agent_n in range(len(self.beliefs['absolute_obs'][agent_id][1])):
            seen_agent = self.beliefs['absolute_obs'][agent_id][1][seen_agent_n]
            seen_agent_id = seen_agent[0]
            #print(f"Considering agent {seen_agent_id} on {agent_id}")
            if seen_agent_id == agent_id:
                #print(f"They're the same! Skipped!")
                continue
            # If that agent's absolute observations have been computed
            if seen_agent_id in self.beliefs['absolute_obs'] and len(self.beliefs['absolute_obs'][seen_agent_id]) != 0:
                #print(f"Agent {seen_agent_id} contains observations to add to {agent_id}")
                # Add the ones that we don't have to ours and add ours to the other one
                self.join_absolute_observations(seen_agent_id, agent_id)
                self.join_absolute_observations(agent_id, seen_agent_id)
                #print(f"Added {seen_agent_id}'s observations to {agent_id}")
            else:
                #print(f"Agent {seen_agent_id} doesn't yet have observations")
                self.compute_absolute_observations(seen_agent_id)
                #print(f"Computed agent {seen_agent_id}'s observations and added them")
                self.join_absolute_observations(seen_agent_id, agent_id)
                self.join_absolute_observations(agent_id, seen_agent_id)

        #print(f"Finished iterations on {agent_id}.")

    def perceive(self):
        # Update agent's beliefs based on the current environment
        # TODO: HANDLE LAST SEEN
        self.beliefs['last_seen'] = self.beliefs.get('prey_positions')

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

        if (self.agent_id == 0):
            print(f"Agent Observations: {self.observation}")
        # print(f"Agent {self.agent_id} Observations: {self.observation[self.agent_id]}")
        # print(f"Agent 1 Observations: {self.observation[1]}")
        # print(f"Agent {self.agent_id} Position: {self.beliefs['agent_position']}")
        # print(f"Agents {self.agent_id} Positions: {self.beliefs['agent_positions']}")
        # print(f"Prey {self.agent_id} Positions: {self.beliefs['prey_positions']}")
        # print(f"Wall {self.agent_id} Positions: {self.beliefs['wall_positions']}")

        # We only get the amount of preys an agent we can't see can see
        agent_id_relative_obs = {}

        # Compute our own absolute observations
        self.compute_absolute_observations(self.agent_id)

        #print(f"Initial absolute observations: {self.beliefs['absolute_obs']}")

        n_agents = len(self.observation)
        # Compute our relative observations (agents not included in absolute observations aren't seen by us)
        for agent_id in range(n_agents):
            if len(self.observation[agent_id]) != 4:
                continue
            if (agent_id not in self.beliefs['absolute_obs']): # if we can't see the agent
                # {agent_id} : [{amount_of_preys}, [{agent_coordenates}]]
                agent_id_relative_obs[agent_id] = [len(self.observation[agent_id][2]), self.observation[agent_id][0][1]]

        self.beliefs['relative_obs'] = agent_id_relative_obs

        # Finish absolute observations to assign cooperations
        for agent_id in range(n_agents):
            if len(self.observation[agent_id]) != 4:
                continue
            self.compute_absolute_observations(agent_id)

        print(f"Absolute Observations on agent {self.agent_id}: {self.beliefs['absolute_obs']}")

        for agent_id in self.conventions:
            if len(self.observation[agent_id]) != 4:
                continue
            print(f"Is agent {self.agent_id} cooperating? {self.cooperating}")
            self.assign_cooperations(agent_id)

        print(f"Cooperations on agent {self.agent_id}: {self.beliefs['cooperations']}")

    def update_desires(self):
        # Update agent's desires based on its beliefs
        if not self.cooperating:
            print(f"Agent {self.agent_id} is not cooperating with any agents")
            # Find the [ids, coords] of agents that don't have a cooperation bond
            non_cooperating_agents = [agent_id for agent_id in self.beliefs['cooperations'] \
                                      if self.beliefs['cooperations'][agent_id][0][0] == None and agent_id != self.agent_id]

            # If all other agents have a cooperation, the agent doesn't desire anything and will move randomly
            if (len(non_cooperating_agents) == 0):
                print(f"Agent {self.agent_id} didn't find any bondless agents to move towards")
                return

            print(f"Found the bondless agents: {non_cooperating_agents}")

            # Find the closest bondless agent
            distances = []
            for agent in non_cooperating_agents:
                other_agent_coords = self.beliefs['observations'][agent][0][1]
                distances.append([other_agent_coords, cityblock(self.beliefs['agent_position'], other_agent_coords)])

            distances.sort(key=lambda x: x[1])

            print(f"Agent {self.agent_id} found a bondless agent at {distances[0][0]}")
            # Move towards the closest agent
            self.desires['desired_location'] = distances[0][0]
        # If we're collaborating and chasing a prey
        elif self.beliefs['cooperations'][self.agent_id][1] != None:
            cooperation_agent_id = self.beliefs['cooperations'][self.agent_id][0][0]
            desired_prey_coords = self.beliefs['cooperations'][self.agent_id][1]
            seen_preys = self.beliefs['absolute_obs'][self.agent_id][2]
            seen_walls = self.beliefs['absolute_obs'][self.agent_id][3]
            print(f"Agent {self.agent_id} is cooperating with agent {cooperation_agent_id} and chasing a prey at {desired_prey_coords}")

            moves = {
                 DOWN: move if (move := self._apply_move(desired_prey_coords, DOWN)) not in seen_walls and move not in seen_preys else None,
                 LEFT: move if (move := self._apply_move(desired_prey_coords, LEFT)) not in seen_walls and move not in seen_preys else None,
                 UP: move if (move := self._apply_move(desired_prey_coords, UP)) not in seen_walls and move not in seen_preys else None,
                 RIGHT: move if (move := self._apply_move(desired_prey_coords, RIGHT)) not in seen_walls and move not in seen_preys else None
            }
            possible_moves = [x for x in moves if moves[x] != None]
            
            # There is only 1 available move, so we race to it
            if len(possible_moves) == 1:
                print(f"Only found one possible move for pair of agents {[self.agent_id, cooperation_agent_id]}")
                self.desires['desired_location'] = moves[possible_moves[0]]
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

            print(f"Our distances: {our_distances}, other_distances: {other_distances}")
            print(f"Our best destination: {moves[our_distances[0][0]]}, other's best destination: {moves[other_distances[0][0]]}")

            if (our_distances[0][0] == other_distances[0][0]):
                print(f"Preferences are the same")
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
                print(f"Agent with lower ID moves to {moves[move1]} and other to {moves[move2]}")
                self.desires['desired_location'] = moves[move1] if order_self < order_other else moves[move2]
            else:
                self.desires['desired_location'] = moves[our_distances[0][0]]
        # We're collaborating, but don't see any preys
        else:
            cooperation_agent_id = self.beliefs['cooperations'][self.agent_id][0][0]
            seen_walls = self.beliefs['absolute_obs'][self.agent_id][3]
            relative_obs = self.beliefs['relative_obs']
            # {agent_id} : [{amount_of_preys}, [{agent_coordenates}]]
            print(f"Agent {self.agent_id} is cooperating with agent {cooperation_agent_id} without any prey")
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
                print(f"Pair of agents {[self.agent_id, cooperation_agent_id]} didn't find any relative observations")
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

                deterministic_index = (our_agent_coords[0] + other_agent_coords[1]) * (our_agent_coords[1] + other_agent_coords[0]) % size
                print(f"Agents picked move {possible_moves[deterministic_index]}")
                self.desires['desired_location'] = moves[possible_moves[deterministic_index]]
            # Move towards the closest agent with observations, preferably at least 2 preys
            else:
                print(f"Pair of agents {[self.agent_id, cooperation_agent_id]} found relative observations, 2: {possible_agents_2}, 1: {possible_agents_1}")
                average_coords = [round((our_agent_coords[0] + other_agent_coords[0])/2), round((our_agent_coords[1] + other_agent_coords[1])/2)]

                iterator = possible_agents_2 if len(possible_agents_2) > 0 else possible_agents_1
                distances = []
                for agent_id in iterator:
                    distances.append([agent_id, cityblock(relative_obs[agent_id][1], average_coords)])
                distances.sort(key=lambda x: x[1])

                print(f"Agents moving towards {relative_obs[distances[0][0]][1]}")
                self.desires['desired_location'] = relative_obs[distances[0][0]][1]

    def deliberation(self):
        # Select intentions based on agent's desires and beliefs
        if 'desired_location' in self.desires:
            print(f"Agent {self.agent_id} will move towards {self.desires['desired_location']}")
            self.intentions['location'] = self.desires['desired_location']
            #print(f"\Location: {self.intentions['location']}\n")
            # we aren't going to cooperate in this step and therefore we will only move to the desired location
        else:
            print(f"Agent {self.agent_id} will move randomly")
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
            if self.intentions['location'] == 'random': # move randomly
                move = np.random.choice(possible_moves)
                print(f"Agent {self.agent_id}'s random move: {move}")
                #print(f"\Move Randomly: {move}\n")
            else: # move to the desired location
                move = self.direction_to_go(self.beliefs['agent_position'], self.intentions['location'], possible_moves)
                print(f"Agent {self.agent_id}'s standard move: {move}")
                #print(f"\Move Location: {move}\n")
        else: # neither is there so move randomly
            print("NO LOCATION INTENTION DEFINED. ERROR.")
            move = np.random.choice(possible_moves)
            print(f"Agent {self.agent_id}'s error move: {move}")
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