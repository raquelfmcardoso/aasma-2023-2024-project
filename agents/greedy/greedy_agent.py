import math
import random
import numpy as np
from scipy.spatial.distance import cityblock

from aasma import Agent

N_ACTIONS = 5
DOWN, LEFT, UP, RIGHT, STAY = range(N_ACTIONS)


class GreedyAgent(Agent):

    """
    A baseline agent for the SimplifiedPredatorPrey environment.
    The greedy agent finds the nearest prey and moves towards it.
    """

    def __init__(self, agent_id):
        super(GreedyAgent, self).__init__(f"Greedy Agent")
        self.agent_id = agent_id
        self.n_actions = N_ACTIONS

    def action(self) -> int:
        # print(f"Agents' observations:", self.observation)
        # print(f"Agent {self.agent_id}'s observation:", self.observation[self.agent_id])
        # print(f"Agent {self.agent_id}'s coordinates:", self.observation[self.agent_id][0])
        # print(f"Agent {self.agent_id}'s observed agents:", self.observation[self.agent_id][1])
        # print(f"Agent {self.agent_id}'s observed preys:", self.observation[self.agent_id][2])
        # print(f"Agent {self.agent_id}'s observed walls:", self.observation[self.agent_id][3])

        agent_position = self.observation[self.agent_id][0][1]
        agent_positions = self.observation[self.agent_id][1]
        prey_positions = self.observation[self.agent_id][2]
        wall_positions = self.observation[self.agent_id][3]
        # We get the complete observations of agents we can see
        agent_id_absolute_obs = {}
        # We only get the amount of preys an agent we can't see can see
        agent_id_relative_obs = {}

        agent_id_absolute_obs[self.agent_id] = self.observation[self.agent_id]
        for agent in agent_positions:
            agent_id_absolute_obs[agent[0]] = self.observation[agent[0]]

        for agent_observation in self.observation:
            if len(agent_observation) != 4:
                continue
            if (agent_observation[0][0] not in agent_id_absolute_obs):
                agent_id_relative_obs[tuple(agent_observation[0][1])] = len(self.observation[agent_observation[0][0]][2])

        moves = {
                 DOWN: move if (move := self._apply_move([agent_position[0], agent_position[1]], DOWN)) not in wall_positions else None,
                 LEFT: move if (move := self._apply_move([agent_position[0], agent_position[1]], LEFT)) not in wall_positions else None,
                 UP: move if (move := self._apply_move([agent_position[0], agent_position[1]], UP)) not in wall_positions else None,
                 RIGHT: move if (move := self._apply_move([agent_position[0], agent_position[1]], RIGHT)) not in wall_positions else None,
                 STAY: [agent_position[0], agent_position[1]]
                }
        # Only account for moves that don't try to move into a wall
        possible_moves = [x for x in moves if moves[x] != None]
        # print("Agent's possible moves:", possible_moves)

        closest_prey = self.closest_prey(agent_position, prey_positions)
        prey_found = closest_prey is not None
        # print("Closest prey:", closest_prey)
        return self.direction_to_go(agent_position, closest_prey, possible_moves) if prey_found else random.choice(possible_moves)

    # ################# #
    # Auxiliary Methods #
    # ################# #

    def direction_to_go(self, agent_position, prey_position, possible_moves):
        """
        Given the position of the agent and the position of a prey,
        returns the action to take in order to close the distance
        """
        distances = np.array(prey_position) - np.array(agent_position)
        # print(f"AGENT: Distance between agent {agent_position} and prey {prey_position}: {distances}")
        abs_distances = np.absolute(distances)
        if abs_distances[0] > abs_distances[1]:
            return self._close_vertically(distances, possible_moves, True)
        elif abs_distances[0] < abs_distances[1]:
            return self._close_horizontally(distances, possible_moves, True)
        else:
            roll = random.uniform(0, 1)
            return self._close_horizontally(distances, possible_moves, True) if roll > 0.5 else self._close_vertically(distances, possible_moves, True)

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

    # ############### #
    # Private Methods #
    # ############### #

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

