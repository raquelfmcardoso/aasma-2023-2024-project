import math
import random
import argparse
import numpy as np
from scipy.spatial.distance import cityblock

from aasma import Agent
from aasma.utils import compare_results
from aasma.wrappers import SingleAgentWrapper
from aasma.simplified_predator_prey import SimplifiedPredatorPrey

from exercise_1_single_random_agent import run_single_agent, RandomAgent

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
        agent_position = self.observation[self.agent_id][0][0], self.observation[self.agent_id][0][1]
        prey_positions = self.observation[self.agent_id][1:]
        closest_prey = self.closest_prey(agent_position, prey_positions)
        prey_found = closest_prey is not None
        print("Closest prey:", closest_prey)
        return self.direction_to_go(agent_position, closest_prey) if prey_found else random.randrange(N_ACTIONS)

    # ################# #
    # Auxiliary Methods #
    # ################# #

    def direction_to_go(self, agent_position, prey_position):
        """
        Given the position of the agent and the position of a prey,
        returns the action to take in order to close the distance
        """
        distances = np.array(prey_position) - np.array(agent_position)
        print(f"AGENT: Distance between agent {agent_position} and prey {prey_position}: {distances}")
        abs_distances = np.absolute(distances)
        if abs_distances[0] > abs_distances[1]:
            print(f"AGENT: Closing horizontally")
            return self._close_vertically(distances)
        elif abs_distances[0] < abs_distances[1]:
            print(f"AGENT: Closing vertically")
            return self._close_horizontally(distances)
        else:
            roll = random.uniform(0, 1)
            return self._close_horizontally(distances) if roll > 0.5 else self._close_vertically(distances)

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

    def _close_horizontally(self, distances):
        if distances[1] > 0:
            return RIGHT
        elif distances[1] < 0:
            return LEFT
        else:
            return STAY

    def _close_vertically(self, distances):
        if distances[0] > 0:
            return DOWN
        elif distances[0] < 0:
            return UP
        else:
            return STAY
        
class GreedyPrey(Agent):

    """
    A baseline prey for the SimplifiedPredatorPrey environment.
    The greedy prey finds the closest agent and moves away from it.
    """

    def __init__(self, prey_id):
        super(GreedyPrey, self).__init__(f"Greedy Prey")
        self.prey_id = prey_id
        self.n_actions = N_ACTIONS

    def action(self) -> int:
        prey_position = self.observation[self.prey_id][0][0], self.observation[self.prey_id][0][1]
        agent_positions = self.observation[self.prey_id][1:]    
        closest_agent = self.closest_agent(prey_position, agent_positions)
        agent_found = closest_agent is not None
        print(f"Closest agent for prey {self.prey_id}: {closest_agent}")
        return self.direction_to_go(prey_position, closest_agent) if agent_found else random.randrange(N_ACTIONS)

    # ################# #
    # Auxiliary Methods #
    # ################# #

    def direction_to_go(self, prey_position, agent_position):
        """
        Given the position of the agent and the position of a prey,
        returns the action to take in order to close the distance
        """
        distances = np.array(agent_position) - np.array(prey_position)
        print(f"PREY: Distance between agent {agent_position} and prey {prey_position}: {distances}")
        abs_distances = np.absolute(distances)
        if abs_distances[0] > abs_distances[1]:
            return self._close_vertically(distances)
        elif abs_distances[0] < abs_distances[1]:
            return self._close_horizontally(distances)
        else:
            roll = random.uniform(0, 1)
            return self._close_horizontally(distances) if roll > 0.5 else self._close_vertically(distances)

    def closest_agent(self, prey_position, agent_positions):
        """
        Given the positions of a prey and a sequence of positions of all agents,
        returns the positions of the closest prey.
        If there are no preys, None is returned instead
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

    def _close_horizontally(self, distances):
        if distances[1] > 0:
            return LEFT
        elif distances[1] < 0:
            return RIGHT
        else:
            return STAY

    def _close_vertically(self, distances):
        if distances[0] > 0:
            return UP
        elif distances[0] < 0:
            return DOWN
        else:
            return STAY

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--episodes", type=int, default=30)
    opt = parser.parse_args()

    # 1 - Setup environment
    environment = SimplifiedPredatorPrey(
        grid_shape=(7, 7),
        n_agents=1, n_preys=1,
        max_steps=100, required_captors=1
    )
    environment = SingleAgentWrapper(environment, agent_id=0)

    # 2 - Setup agents
    agents = [
        RandomAgent(environment.action_space.n),
        GreedyAgent(agent_id=0, n_agents=1)
    ]

    # 3 - Evaluate agents
    results = {}
    for agent in agents:
        result = run_single_agent(environment, agent, opt.episodes)
        results[agent.name] = result

    # 4 - Compare results
    compare_results(results, title="Agents on 'Predator Prey' Environment", colors=["orange", "green"])

