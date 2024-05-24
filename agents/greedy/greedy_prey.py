import math
import random
import numpy as np
from scipy.spatial.distance import cityblock

from aasma import Agent

N_ACTIONS = 5
DOWN, LEFT, UP, RIGHT, STAY = range(N_ACTIONS)

class GreedyPrey(Agent):

    """
    Greeedy prey for the Tigers vs Deer environment.
    The greedy prey finds the closest agent and moves away from it.
    """

    def __init__(self, prey_id):
        super(GreedyPrey, self).__init__(f"Greedy Prey")
        self.prey_id = prey_id
        self.n_actions = N_ACTIONS

    def action(self) -> int:
        prey_position = self.observation[self.prey_id][0][1]
        agent_positions = self.observation[self.prey_id][1]
        prey_positions = self.observation[self.prey_id][2]
        wall_positions = self.observation[self.prey_id][3]
        # We get the complete observations of preys of the same species we can see
        prey_id_absolute_obs = {}

        prey_id_absolute_obs[self.prey_id] = self.observation[self.prey_id]
        for prey in prey_positions:
            prey_id_absolute_obs[prey[0]] = self.observation[prey[0]]

        nearby_agents = agent_positions
        for observation in prey_id_absolute_obs:
            for pos in prey_id_absolute_obs[observation][1]:
                if pos not in nearby_agents:
                    nearby_agents.append(pos)

        moves = {
                 DOWN: move if (move := self._apply_move([prey_position[0], prey_position[1]], DOWN)) not in wall_positions else None,
                 LEFT: move if (move := self._apply_move([prey_position[0], prey_position[1]], LEFT)) not in wall_positions else None,
                 UP: move if (move := self._apply_move([prey_position[0], prey_position[1]], UP)) not in wall_positions else None,
                 RIGHT: move if (move := self._apply_move([prey_position[0], prey_position[1]], RIGHT)) not in wall_positions else None,
                 STAY: [prey_position[0], prey_position[1]]
                }
        possible_moves = [x for x in moves if moves[x] != None]
        
        closest_agent = self.closest_agent(prey_position, nearby_agents)
        agent_found = closest_agent is not None
        return self.direction_to_go(prey_position, closest_agent, possible_moves) if agent_found else random.choice(possible_moves)

    def run(self):
        return self.action()

    # ################# #
    # Auxiliary Methods #
    # ################# #

    def direction_to_go(self, prey_position, agent_position, possible_moves):
        """
        Given the position of the prey, the position of a predator, and the possible moves,
        returns the action to take in order to maximize the distance
        """
        distances = np.array(prey_position) - np.array(agent_position)
        abs_distances = np.absolute(distances)
        if abs_distances[0] > abs_distances[1]:
            return self._close_vertically(distances, possible_moves, True)
        elif abs_distances[0] < abs_distances[1]:
            return self._close_horizontally(distances, possible_moves, True)
        else:
            roll = random.uniform(0, 1)
            return self._close_horizontally(distances, possible_moves, True) if roll > 0.5 else self._close_vertically(distances, possible_moves, True)

    def closest_agent(self, prey_position, agent_positions):
        """
        Given the positions of a prey and a sequence of positions of all predators,
        returns the positions of the closest predator.
        If there are no predators, None is returned instead
        """
        min = math.inf
        closest_agent_position = None
        n_agents = len(agent_positions)
        for p in range(n_agents):
            agent_position = agent_positions[p][0], agent_positions[p][1]
            distance = cityblock(prey_position, agent_position)
            if distance < min:
                min = distance
                closest_agent_position = agent_position
        return closest_agent_position

    # ############### #
    # Private Methods #
    # ############### #

    def _apply_move(self, prey_position, move):
        if move == RIGHT:
            return [prey_position[0], prey_position[1] + 1]
        elif move == LEFT:
            return [prey_position[0], prey_position[1] - 1]
        elif move == UP:
            return [prey_position[0] - 1, prey_position[1]]
        elif move == DOWN:
            return [prey_position[0] + 1, prey_position[1]]
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