import copy
import logging
import random

import numpy as np

logger = logging.getLogger(__name__)

from PIL import ImageColor
import gym
from gym import spaces
from gym.utils import seeding

from ma_gym.envs.utils.action_space import MultiAgentActionSpace
from ma_gym.envs.utils.draw import draw_grid, fill_cell, draw_circle, write_cell_text
from ma_gym.envs.utils.observation_space import MultiAgentObservationSpace

class SimplifiedPredatorPrey(gym.Env):

    """A simplified version of ma_gym.envs.predator_prey.predator_prey.PredatorPrey
    Observations do not take into account the nearest cells and an extra parameter (required_captors) was added

    See Also
    --------
    ma_gym.envs.predator_prey

    """

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, grid_shape=(5, 5), n_agents=2, n_preys=1, n_preys2=1, prey_move_probs=(0.175, 0.175, 0.175, 0.175, 0.3),
                 full_observable=False, penalty=-0.5, step_cost=-0.01, prey_capture_reward=5, max_steps=100, required_captors=2, n_obstacles=10):
    def __init__(self, grid_shape=(5, 5), n_agents=2, n_preys=1, n_preys2=1, prey_move_probs=(0.175, 0.175, 0.175, 0.175, 0.3),
                 full_observable=False, penalty=-0.5, step_cost=-0.01, prey_capture_reward=5, max_steps=100, required_captors=2, n_obstacles=10):
        self._grid_shape = grid_shape
        self.n_agents = n_agents
        self.n_preys = n_preys
        self.n_preys2 = n_preys2
        self.n_preys2 = n_preys2
        self._max_steps = max_steps
        self._step_count = None
        self._penalty = penalty
        self._step_cost = step_cost
        self._prey_capture_reward = prey_capture_reward
        self._view_mask = (5, 5) # para both prey e predator
        self._required_captors = required_captors
        self._n_obstacles = n_obstacles
        self._n_obstacles = n_obstacles

        self.action_space = MultiAgentActionSpace([spaces.Discrete(5) for _ in range(self.n_agents)])
        self.agent_action_space = MultiAgentActionSpace([spaces.Discrete(5) for _ in range(self.n_agents)])
        self.prey_action_space = MultiAgentActionSpace([spaces.Discrete(5) for _ in range(self.n_preys)])
        self.prey2_action_space = MultiAgentActionSpace([spaces.Discrete(5) for _ in range(self.n_preys2)])
        self.prey2_action_space = MultiAgentActionSpace([spaces.Discrete(5) for _ in range(self.n_preys2)])
        self.agent_pos = {_: None for _ in range(self.n_agents)}
        self.prey_pos = {_: None for _ in range(self.n_preys)}
        self.prey2_pos = {_: None for _ in range(self.n_preys2)}
        self.prey2_pos = {_: None for _ in range(self.n_preys2)}
        self._prey_alive = None
        self._prey_alive2 = None
        self._prey_alive2 = None

        self._base_grid = self.__create_grid()  # with no agents
        self._full_obs = copy.copy(self._base_grid)
        self._full_obs = copy.copy(self._base_grid)
        self._agent_dones = [False for _ in range(self.n_agents)]
        self._prey_move_probs = prey_move_probs
        self._prey2_move_probs = prey_move_probs
        self._prey2_move_probs = prey_move_probs
        self.viewer = None
        self.full_observable = full_observable

        # agent pos (2), prey (25), step (1)
        mask_size = np.prod(self._view_mask)
        self._obs_high = np.array([1., 1.] + [1.] * mask_size + [1.0], dtype=np.float32)
        self._obs_low = np.array([0., 0.] + [0.] * mask_size + [0.0], dtype=np.float32)
        if self.full_observable:
            self._obs_high = np.tile(self._obs_high, self.n_agents)
            self._obs_low = np.tile(self._obs_low, self.n_agents)
        self.observation_space = MultiAgentObservationSpace(
            [spaces.Box(self._obs_low, self._obs_high) for _ in range(self.n_agents)])

        self._total_episode_reward = None
        self.seed()

    def deepcopy(self, grid):
        new_grid = []
        for row in grid:
            new_row = []
            for y in row:
                new_row.append(y)
            new_grid.append(new_row)
        return new_grid

    def deepcopy(self, grid):
        new_grid = []
        for row in grid:
            new_row = []
            for y in row:
                new_row.append(y)
            new_grid.append(new_row)
        return new_grid

    def simplified_features(self):

        current_grid = np.array(self._full_obs)

        agent_pos = []
        for agent_id in range(self.n_agents):
            tag = f"A{agent_id + 1}"
            row, col = np.where(current_grid == tag)
            row = row[0]
            col = col[0]
            agent_pos.append((col, row))

        prey_pos = []
        for prey_id in range(self.n_preys):
            if self._prey_alive[prey_id]:
                tag = f"P{prey_id + 1}"
                row, col = np.where(current_grid == tag)
                row = row[0]
                col = col[0]
                prey_pos.append((col, row))

        prey_pos2 = []
        for prey_id in range(self.n_preys2):
            if self._prey_alive2[prey_id]:
                tag = f"P2{prey_id + 1}"
                row, col = np.where(current_grid == tag)
                row = row[0]
                col = col[0]
                prey_pos2.append((col, row))

        features = np.array(agent_pos + prey_pos + prey_pos2).reshape(-1)

        return features

    def reset(self):
        self._total_episode_reward = [0 for _ in range(self.n_agents)]
        self.agent_pos = {}
        self.prey_pos = {}
        self.prey2_pos = {}
        self.prey2_pos = {}

        self.__init_full_obs()
        self._step_count = 0
        self._agent_dones = [False for _ in range(self.n_agents)]
        self._prey_alive = [True for _ in range(self.n_preys)]
        self._prey_alive2 = [True for _ in range(self.n_preys2)]
        self._prey_alive2 = [True for _ in range(self.n_preys2)]

        #self.get_agent_obs()
        #self.get_prey_obs()
        #self.get_prey2_obs()
        #return [self.simplified_features() for _ in range(self.n_agents)]
        return [self.get_agent_obs(), self.get_prey_obs(), self.get_prey2_obs()]

    def step(self, agents_action, preys_action, preys2_action):
    def step(self, agents_action, preys_action, preys2_action):
        self._step_count += 1
        rewards = [self._step_cost for _ in range(self.n_agents)]

        for prey_i, action in enumerate(preys_action):
            if self._prey_alive[prey_i]:
                predator_neighbour_count, n_i = self._neighbour_agents(self.prey_pos[prey_i])
                if predator_neighbour_count >= self._required_captors:
                    _reward = self._prey_capture_reward
                    self._prey_alive[prey_i] = (predator_neighbour_count < self._required_captors)

                    for agent_i in range(self.n_agents):
                        rewards[agent_i] += _reward
                
                self.__update_prey_pos(prey_i, action)

        for prey_i, action in enumerate(preys2_action):
            if self._prey_alive2[prey_i]:
                predator_neighbour_count, n_i = self._neighbour_agents(self.prey2_pos[prey_i])
                if predator_neighbour_count >= self._required_captors:
                    _reward = self._prey_capture_reward
                    self._prey_alive2[prey_i] = (predator_neighbour_count < self._required_captors)

                    for agent_i in range(self.n_agents):
                        rewards[agent_i] += _reward
                
                self.__update_prey2_pos(prey_i, action)

        for agent_i, action in enumerate(agents_action):
            if not (self._agent_dones[agent_i]):
                self.__update_agent_pos(agent_i, action)
                
        if (self._step_count >= self._max_steps) or (True not in self._prey_alive) or (True not in self._prey_alive2):

        for prey_i, action in enumerate(preys2_action):
            if self._prey_alive2[prey_i]:
                predator_neighbour_count, n_i = self._neighbour_agents(self.prey2_pos[prey_i])
                if predator_neighbour_count >= self._required_captors:
                    _reward = self._prey_capture_reward
                    self._prey_alive2[prey_i] = (predator_neighbour_count < self._required_captors)

                    for agent_i in range(self.n_agents):
                        rewards[agent_i] += _reward
                
                self.__update_prey2_pos(prey_i, action)

        for agent_i, action in enumerate(agents_action):
            if not (self._agent_dones[agent_i]):
                self.__update_agent_pos(agent_i, action)
                
        if (self._step_count >= self._max_steps) or (True not in self._prey_alive) or (True not in self._prey_alive2):
            for i in range(self.n_agents):
                self._agent_dones[i] = True

        for i in range(self.n_agents):
            self._total_episode_reward[i] += rewards[i]

        #self.get_agent_obs()
        #self.get_prey_obs()
        #self.get_prey2_obs()
        #return [self.simplified_features() for _ in range(self.n_agents)], rewards, self._agent_dones, {'prey_alive': self._prey_alive, 'prey_alive2': self._prey_alive2}
        return [self.get_agent_obs(), self.get_prey_obs(), self.get_prey2_obs()], rewards, self._agent_dones, {'prey_alive': self._prey_alive, 'prey_alive2': self._prey_alive2}

    def get_action_meanings(self, agent_i=None):
        if agent_i is not None:
            assert agent_i <= self.n_agents
            return [ACTION_MEANING[i] for i in range(self.action_space[agent_i].n)]
        else:
            return [[ACTION_MEANING[i] for i in range(ac.n)] for ac in self.action_space]

    def agent_action_space_sample(self):
        return [agent_action_space.sample() for agent_action_space in self.agent_action_space]
    
    def prey_action_space_sample(self):
        return [prey_action_space.sample() for prey_action_space in self.prey_action_space]
    
    def prey2_action_space_sample(self):
        return [prey2_action_space.sample() for prey2_action_space in self.prey2_action_space]
    
    def place_l_wall(self, x, y, orientation, grid):
        if (self.is_valid((x, y))):
                grid[x][y] = PRE_IDS['wall']
        if orientation == 'up':
            if (self.is_valid((x, y + 1))):
                grid[x][y + 1] = PRE_IDS['wall']
            if (self.is_valid((x + 1, y))):
                grid[x + 1][y] = PRE_IDS['wall']
        elif orientation == 'down':
            if (self.is_valid((x, y - 1))):
                grid[x][y - 1] = PRE_IDS['wall']
            if (self.is_valid((x, y + 1))):
                grid[x][y + 1] = PRE_IDS['wall']
        elif orientation == 'left':
            if (self.is_valid((x - 1, y))):
                grid[x - 1][y] = PRE_IDS['wall']
            if (self.is_valid((x, y - 1))):
                grid[x][y - 1] = PRE_IDS['wall']
        elif orientation == 'right':
            if (self.is_valid((x + 1, y))):
                grid[x + 1][y] = PRE_IDS['wall']
            if (self.is_valid((x, y + 1))):
                grid[x][y + 1] = PRE_IDS['wall']

    def place_straight_line(self, x, y, direction, length, grid):
        if direction == 'horizontal':
            for i in range(length):
                if (self.is_valid((x, y + i))):
                    grid[x][y + i] = PRE_IDS['wall']
        elif direction == 'vertical':
            for i in range(length):
                if (self.is_valid((x + i, y))):
                    grid[x + i][y] = PRE_IDS['wall']

    def generate_walls(self, grid, n_obstacles=10):
        for _ in range(n_obstacles):
            obstacle_type = random.choice(['L', 'line'])
            x = random.randint(0, self._grid_shape[0] - 1)
            y = random.randint(0, self._grid_shape[1] - 1)

            if obstacle_type == 'L':
                orientation = random.choice(['up', 'down', 'left', 'right'])
                self.place_l_wall(x, y, orientation, grid)
            elif obstacle_type == 'line':
                direction = random.choice(['horizontal', 'vertical'])
                length = random.randint(2, 3)
                self.place_straight_line(x, y, direction, length, grid)
    
    def prey2_action_space_sample(self):
        return [prey2_action_space.sample() for prey2_action_space in self.prey2_action_space]
    
    def place_l_wall(self, x, y, orientation, grid):
        if (self.is_valid((x, y))):
                grid[x][y] = PRE_IDS['wall']
        if orientation == 'up':
            if (self.is_valid((x, y + 1))):
                grid[x][y + 1] = PRE_IDS['wall']
            if (self.is_valid((x + 1, y))):
                grid[x + 1][y] = PRE_IDS['wall']
        elif orientation == 'down':
            if (self.is_valid((x, y - 1))):
                grid[x][y - 1] = PRE_IDS['wall']
            if (self.is_valid((x, y + 1))):
                grid[x][y + 1] = PRE_IDS['wall']
        elif orientation == 'left':
            if (self.is_valid((x - 1, y))):
                grid[x - 1][y] = PRE_IDS['wall']
            if (self.is_valid((x, y - 1))):
                grid[x][y - 1] = PRE_IDS['wall']
        elif orientation == 'right':
            if (self.is_valid((x + 1, y))):
                grid[x + 1][y] = PRE_IDS['wall']
            if (self.is_valid((x, y + 1))):
                grid[x][y + 1] = PRE_IDS['wall']

    def place_straight_line(self, x, y, direction, length, grid):
        if direction == 'horizontal':
            for i in range(length):
                if (self.is_valid((x, y + i))):
                    grid[x][y + i] = PRE_IDS['wall']
        elif direction == 'vertical':
            for i in range(length):
                if (self.is_valid((x + i, y))):
                    grid[x + i][y] = PRE_IDS['wall']

    def generate_walls(self, grid, n_obstacles=10):
        for _ in range(n_obstacles):
            obstacle_type = random.choice(['L', 'line'])
            x = random.randint(0, self._grid_shape[0] - 1)
            y = random.randint(0, self._grid_shape[1] - 1)

            if obstacle_type == 'L':
                orientation = random.choice(['up', 'down', 'left', 'right'])
                self.place_l_wall(x, y, orientation, grid)
            elif obstacle_type == 'line':
                direction = random.choice(['horizontal', 'vertical'])
                length = random.randint(2, 3)
                self.place_straight_line(x, y, direction, length, grid)

    def __draw_base_img(self):
        self._base_img = draw_grid(self._grid_shape[0], self._grid_shape[1], cell_size=CELL_SIZE, fill='white')
        for x in range(self._grid_shape[0]):
            for y in range(self._grid_shape[1]):
                if self._full_obs[x][y] == PRE_IDS['wall']:
                    fill_cell(self._base_img, [x, y], cell_size=CELL_SIZE, fill=WALL_COLOR, margin=0.1)
        for x in range(self._grid_shape[0]):
            for y in range(self._grid_shape[1]):
                if self._full_obs[x][y] == PRE_IDS['wall']:
                    fill_cell(self._base_img, [x, y], cell_size=CELL_SIZE, fill=WALL_COLOR, margin=0.1)

    def __create_grid(self):
        _grid = [[PRE_IDS['empty'] for y in range(self._grid_shape[1])] for x in range(self._grid_shape[0])]
        self.generate_walls(_grid, n_obstacles = self._n_obstacles)
        _grid = [[PRE_IDS['empty'] for y in range(self._grid_shape[1])] for x in range(self._grid_shape[0])]
        self.generate_walls(_grid, n_obstacles = self._n_obstacles)
        return _grid

    def __init_full_obs(self):
        self._full_obs = self.__create_grid()

        for agent_i in range(self.n_agents):
            while True:
                pos = [self.np_random.randint(0, self._grid_shape[0] - 1),
                       self.np_random.randint(0, self._grid_shape[1] - 1)]
                if self._is_cell_vacant(pos):
                    self.agent_pos[agent_i] = pos
                    break
            self.__update_agent_view(agent_i)

        for prey_i in range(self.n_preys):
            while True:
                pos = [self.np_random.randint(0, self._grid_shape[0] - 1),
                       self.np_random.randint(0, self._grid_shape[1] - 1)]
                if self._is_cell_vacant(pos) and (self._neighbour_agents(pos)[0] == 0):
                    self.prey_pos[prey_i] = pos
                    break
            self.__update_prey_view(prey_i)

        for prey_i in range(self.n_preys2):
            while True:
                pos = [self.np_random.randint(0, self._grid_shape[0] - 1),
                       self.np_random.randint(0, self._grid_shape[1] - 1)]
                if self._is_cell_vacant(pos) and (self._neighbour_agents(pos)[0] == 0):
                    self.prey2_pos[prey_i] = pos
                    break
            self.__update_prey2_view(prey_i)

        for prey_i in range(self.n_preys2):
            while True:
                pos = [self.np_random.randint(0, self._grid_shape[0] - 1),
                       self.np_random.randint(0, self._grid_shape[1] - 1)]
                if self._is_cell_vacant(pos) and (self._neighbour_agents(pos)[0] == 0):
                    self.prey2_pos[prey_i] = pos
                    break
            self.__update_prey2_view(prey_i)

        self.__draw_base_img()

    def get_agent_obs(self):
        _obs = []
        for agent_i in range(self.n_agents):
            pos = self.agent_pos[agent_i]
            #_agent_i_obs = [pos[0] / (self._grid_shape[0] - 1), pos[1] / (self._grid_shape[1] - 1)]  # coordinates
            _agent_i_pos = []
            _agent_i_pos.append(pos)

            # check if prey is in the view area
            #_prey_pos = np.zeros(self._view_mask)  # prey location in neighbour
            for row in range(max(0, pos[0] - 10), min(pos[0] + 10 + 1, self._grid_shape[0])):
                for col in range(max(0, pos[1] - 10), min(pos[1] + 10 + 1, self._grid_shape[1])):
                    if (PRE_IDS['prey'] or PRE_IDS['prey2']) in self._full_obs[row][col]:
                        # Horizontal verification
                        if (row == pos[0] and col > pos[1] and PRE_IDS['wall'] in self._full_obs[row][(pos[1] + 1):(col + 1)]):
                            print("Horizontal wall found on the right of " , agent_i)
                            break
                        elif (row == pos[0] and col < pos[1] and PRE_IDS['wall'] in self._full_obs[row][col:pos[1]]):
                            print("Horizontal wall found on the left of " , agent_i)
                            continue
                        # Vertical verification
                        elif (col == pos[1] and row > pos[0] and any(x == PRE_IDS['wall'] for i in range(pos[0] + 1, row) for x in self._full_obs[i][col])):
                            print("Vertical wall found below " , agent_i)
                            continue
                        elif (col == pos[1] and row < pos[0] and any(x == PRE_IDS['wall'] for i in range(row, pos[0]) for x in self._full_obs[i][col])):
                            print("Vertical wall found above " , agent_i)
                            continue
                        # Diagonal verification
                        if abs(row - pos[0]) == abs(col - pos[1]):
                            if (row - pos[0] > 0):
                                print("Prey is below the agent")
                                if (col - pos[0] > 0 and self.verify_diagonal((-1, -1), pos, [row, col])):
                                    print("Diagonal wall found on the bottom right of " , agent_i)
                                    continue
                                elif (col - pos[0] < 0 and self.verify_diagonal((-1, 1), pos, [row, col])):
                                    print("Diagonal wall found on the bottom left of " , agent_i)
                                    continue
                            elif (row - pos[0] < 0):
                                print("Prey is above the agent")
                                if (col - pos[0] > 0 and self.verify_diagonal((1, -1), pos, [row, col])):
                                    print("Diagonal wall found on the top right of " , agent_i)
                                    continue
                                elif (col - pos[0] < 0 and self.verify_diagonal((1, 1), pos, [row, col])):
                                    print("Diagonal wall found on the top left of " , agent_i)
                                    continue
                        #_prey_pos[row - (pos[0] - 2), col - (pos[1] - 2)] = 1  # get relative position for the prey loc.
                        _agent_i_pos.append([row, col])

            #_agent_i_obs += _prey_pos.flatten().tolist()  # adding prey pos in observable area
            #_agent_i_obs += [self._step_count / self._max_steps]  # adding time
            _obs.append(_agent_i_pos)

        if self.full_observable:
            _obs = np.array(_obs).flatten().tolist()
            _obs = [_obs for _ in range(self.n_agents)]
        return _obs

    def get_prey_obs(self):
        _obs = []
        for prey_i in range(self.n_preys):
            pos = self.prey_pos[prey_i]
            #_prey_i_obs = [pos[0] / (self._grid_shape[0] - 1), pos[1] / (self._grid_shape[1] - 1)]  # coordinates
            _prey_i_pos = []
            _prey_i_pos.append(pos)

            # check if agent is in the view area
            #_agent_pos = np.zeros(self._view_mask)  # agent location in neighbour
            for row in range(max(0, pos[0] - 10), min(pos[0] + 10 + 1, self._grid_shape[0])):
                for col in range(max(0, pos[1] - 10), min(pos[1] + 10 + 1, self._grid_shape[1])):
                    if PRE_IDS['agent'] in self._full_obs[row][col]:
                        # Horizontal verification
                        if (row == pos[0] and col > pos[1] and PRE_IDS['wall'] in self._full_obs[row][(pos[1] + 1):(col + 1)]):
                            print("Horizontal wall found on the right of " , prey_i)
                            break
                        elif (row == pos[0] and col < pos[1] and PRE_IDS['wall'] in self._full_obs[row][col:pos[1]]):
                            print("Horizontal wall found on the left of " , prey_i)
                            continue
                        # Vertical verification
                        elif (col == pos[1] and row > pos[0] and any(x == PRE_IDS['wall'] for i in range(pos[0] + 1, row) for x in self._full_obs[i][col])):
                            print("Vertical wall found below " , prey_i)
                            continue
                        elif (col == pos[1] and row < pos[0] and any(x == PRE_IDS['wall'] for i in range(row, pos[0]) for x in self._full_obs[i][col])):
                            print("Vertical wall found above " , prey_i)
                            continue
                        # Diagonal verification
                        if abs(row - pos[0]) == abs(col - pos[1]):
                            if (row - pos[0] > 0):
                                print("Agent is below the prey")
                                if (col - pos[0] > 0 and self.verify_diagonal((-1, -1), pos, [row, col])):
                                    print("Diagonal wall found on the bottom right of " , prey_i)
                                    continue
                                elif (col - pos[0] < 0 and self.verify_diagonal((-1, 1), pos, [row, col])):
                                    print("Diagonal wall found on the bottom left of " , prey_i)
                                    continue
                            elif (row - pos[0] < 0):
                                print("Agent is above the prey")
                                if (col - pos[0] > 0 and self.verify_diagonal((1, -1), pos, [row, col])):
                                    print("Diagonal wall found on the top right of " , prey_i)
                                    continue
                                elif (col - pos[0] < 0 and self.verify_diagonal((1, 1), pos, [row, col])):
                                    print("Diagonal wall found on the top left " , prey_i)
                                    continue
                        #_agent_pos[row - (pos[0] - 2), col - (pos[1] - 2)] = 1  # get relative position for the prey loc.
                        _prey_i_pos.append([row, col])

            #_prey_i_obs += _agent_pos.flatten().tolist()  # adding agent pos in observable area
            #_prey_i_obs += [self._step_count / self._max_steps]  # adding time
            _obs.append(_prey_i_pos)

        if self.full_observable:
            _obs = np.array(_obs).flatten().tolist()
            _obs = [_obs for _ in range(self.n_preys)]
        return _obs
    
    def get_prey2_obs(self):
        _obs = []
        for prey_i in range(self.n_preys2):
            pos = self.prey2_pos[prey_i]
            #_prey_i_obs = [pos[0] / (self._grid_shape[0] - 1), pos[1] / (self._grid_shape[1] - 1)]
            _prey_i_pos = []
            _prey_i_pos.append(pos)

            #_agent_pos = np.zeros(self._view_mask)
            for row in range(max(0, pos[0] - 10), min(pos[0] + 10 + 1, self._grid_shape[0])):
                for col in range(max(0, pos[1] - 10), min(pos[1] + 10 + 1, self._grid_shape[1])):
                    if PRE_IDS['agent'] in self._full_obs[row][col]:
                        # Horizontal verification
                        if (row == pos[0] and col > pos[1] and PRE_IDS['wall'] in self._full_obs[row][(pos[1] + 1):(col + 1)]):
                            print("Horizontal wall found on the right of " , prey_i)
                            break
                        elif (row == pos[0] and col < pos[1] and PRE_IDS['wall'] in self._full_obs[row][col:pos[1]]):
                            print("Horizontal wall found on the left of " , prey_i)
                            continue
                        # Vertical verification
                        elif (col == pos[1] and row > pos[0] and any(x == PRE_IDS['wall'] for i in range(pos[0] + 1, row) for x in self._full_obs[i][col])):
                            print("Vertical wall found below " , prey_i)
                            continue
                        elif (col == pos[1] and row < pos[0] and any(x == PRE_IDS['wall'] for i in range(row, pos[0]) for x in self._full_obs[i][col])):
                            print("Vertical wall found above " , prey_i)
                            continue
                        # Diagonal verification
                        if abs(row - pos[0]) == abs(col - pos[1]):
                            if (row - pos[0] > 0):
                                print("Agent is below the prey")
                                if (col - pos[0] > 0 and self.verify_diagonal((-1, -1), pos, [row, col])):
                                    print("Diagonal wall found on the bottom right of " , prey_i)
                                    continue
                                elif (col - pos[0] < 0 and self.verify_diagonal((-1, 1), pos, [row, col])):
                                    print("Diagonal wall found on the bottom left of " , prey_i)
                                    continue
                            elif (row - pos[0] < 0):
                                print("Agent is above the prey")
                                if (col - pos[0] > 0 and self.verify_diagonal((1, -1), pos, [row, col])):
                                    print("Diagonal wall found on the top right of " , prey_i)
                                    continue
                                elif (col - pos[0] < 0 and self.verify_diagonal((1, 1), pos, [row, col])):
                                    print("Diagonal wall found on the top left " , prey_i)
                                    continue
                        _prey_i_pos.append([row, col])

            #_prey_i_obs += _agent_pos.flatten().tolist()
            #_prey_i_obs += [self._step_count / self._max_steps]
            _obs.append(_prey_i_pos)
        
        if self.full_observable:
            _obs = np.array(_obs).flatten().tolist()
            _obs = [_obs for _ in range(self.n_preys2)]
        return _obs

    def verify_diagonal(self, iterator: tuple, my_pos: list, other_pos: list):
        x, y = other_pos
        while (0 <= x < self._grid_shape[0] and 0 <= y < self._grid_shape[1]) and (x != my_pos[0] and y != my_pos[1]):
            if self._full_obs[x][y] == PRE_IDS['wall']:
                return True
            x += iterator[0]
            y += iterator[1]
        return False

    def __wall_exists(self, pos):
        row, col = pos
        return PRE_IDS['wall'] in self._base_grid[row, col]

    def is_valid(self, pos):
        return (0 <= pos[0] < self._grid_shape[0]) and (0 <= pos[1] < self._grid_shape[1])

    def _is_cell_vacant(self, pos):
        return self.is_valid(pos) and (self._full_obs[pos[0]][pos[1]] == PRE_IDS['empty'])

    def __update_agent_pos(self, agent_i, move):

        curr_pos = copy.copy(self.agent_pos[agent_i])
        next_pos = None
        if move == 0:  # down
            next_pos = [curr_pos[0] + 1, curr_pos[1]]
        elif move == 1:  # left
            next_pos = [curr_pos[0], curr_pos[1] - 1]
        elif move == 2:  # up
            next_pos = [curr_pos[0] - 1, curr_pos[1]]
        elif move == 3:  # right
            next_pos = [curr_pos[0], curr_pos[1] + 1]
        elif move == 4:  # no-op
            pass
        else:
            raise Exception('Action Not found!')

        if next_pos is not None and self._is_cell_vacant(next_pos):
            self.agent_pos[agent_i] = next_pos
            self._full_obs[curr_pos[0]][curr_pos[1]] = PRE_IDS['empty']
            self.__update_agent_view(agent_i)

    def __next_pos(self, curr_pos, move):
        if move == 0:  # down
            next_pos = [curr_pos[0] + 1, curr_pos[1]]
        elif move == 1:  # left
            next_pos = [curr_pos[0], curr_pos[1] - 1]
        elif move == 2:  # up
            next_pos = [curr_pos[0] - 1, curr_pos[1]]
        elif move == 3:  # right
            next_pos = [curr_pos[0], curr_pos[1] + 1]
        elif move == 4:  # no-op
            next_pos = curr_pos
        return next_pos

    def __update_prey_pos(self, prey_i, move):
        curr_pos = copy.copy(self.prey_pos[prey_i])
        if self._prey_alive[prey_i]:
            next_pos = None
            if move == 0:  # down
                next_pos = [curr_pos[0] + 1, curr_pos[1]]
            elif move == 1:  # left
                next_pos = [curr_pos[0], curr_pos[1] - 1]
            elif move == 2:  # up
                next_pos = [curr_pos[0] - 1, curr_pos[1]]
            elif move == 3:  # right
                next_pos = [curr_pos[0], curr_pos[1] + 1]
            elif move == 4:  # no-op
                pass
            else:
                raise Exception('Action Not found!')

            if next_pos is not None and self._is_cell_vacant(next_pos):
                self.prey_pos[prey_i] = next_pos
                self._full_obs[curr_pos[0]][curr_pos[1]] = PRE_IDS['empty']
                self.__update_prey_view(prey_i)
            else:
                # print('pos not updated')
                pass
        else:
            self._full_obs[curr_pos[0]][curr_pos[1]] = PRE_IDS['empty']

    def __update_prey2_pos(self, prey_i, move):
        curr_pos = copy.copy(self.prey2_pos[prey_i])
        if self._prey_alive2[prey_i]:
            next_pos = None
            if move == 0:
                next_pos = [curr_pos[0] + 1, curr_pos[1]]
            elif move == 1:
                next_pos = [curr_pos[0], curr_pos[1] - 1]
            elif move == 2:
                next_pos = [curr_pos[0] - 1, curr_pos[1]]
            elif move == 3:
                next_pos = [curr_pos[0], curr_pos[1] + 1]
            elif move == 4:
                pass
            else:
                raise Exception('Action Not found!')
            
            if next_pos is not None and self._is_cell_vacant(next_pos):
                self.prey2_pos[prey_i] = next_pos
                self._full_obs[curr_pos[0]][curr_pos[1]] = PRE_IDS['empty']
                self.__update_prey2_view(prey_i)

    def __update_prey2_pos(self, prey_i, move):
        curr_pos = copy.copy(self.prey2_pos[prey_i])
        if self._prey_alive2[prey_i]:
            next_pos = None
            if move == 0:
                next_pos = [curr_pos[0] + 1, curr_pos[1]]
            elif move == 1:
                next_pos = [curr_pos[0], curr_pos[1] - 1]
            elif move == 2:
                next_pos = [curr_pos[0] - 1, curr_pos[1]]
            elif move == 3:
                next_pos = [curr_pos[0], curr_pos[1] + 1]
            elif move == 4:
                pass
            else:
                raise Exception('Action Not found!')
            
            if next_pos is not None and self._is_cell_vacant(next_pos):
                self.prey2_pos[prey_i] = next_pos
                self._full_obs[curr_pos[0]][curr_pos[1]] = PRE_IDS['empty']
                self.__update_prey2_view(prey_i)

    def __update_agent_view(self, agent_i):
        self._full_obs[self.agent_pos[agent_i][0]][self.agent_pos[agent_i][1]] = PRE_IDS['agent'] + str(agent_i + 1)

    def __update_prey_view(self, prey_i):
        self._full_obs[self.prey_pos[prey_i][0]][self.prey_pos[prey_i][1]] = PRE_IDS['prey'] + str(prey_i + 1)

    def __update_prey2_view(self, prey_i):
        self._full_obs[self.prey2_pos[prey_i][0]][self.prey2_pos[prey_i][1]] = PRE_IDS['prey'] + str(prey_i + 1)


    def __update_prey2_view(self, prey_i):
        self._full_obs[self.prey2_pos[prey_i][0]][self.prey2_pos[prey_i][1]] = PRE_IDS['prey'] + str(prey_i + 1)


    def _neighbour_agents(self, pos):
        # check if agent is in neighbour
        _count = 0
        neighbours_xy = []
        if self.is_valid([pos[0] + 1, pos[1]]) and PRE_IDS['agent'] in self._full_obs[pos[0] + 1][pos[1]]:
            _count += 1
            neighbours_xy.append([pos[0] + 1, pos[1]])
        if self.is_valid([pos[0] - 1, pos[1]]) and PRE_IDS['agent'] in self._full_obs[pos[0] - 1][pos[1]]:
            _count += 1
            neighbours_xy.append([pos[0] - 1, pos[1]])
        if self.is_valid([pos[0], pos[1] + 1]) and PRE_IDS['agent'] in self._full_obs[pos[0]][pos[1] + 1]:
            _count += 1
            neighbours_xy.append([pos[0], pos[1] + 1])
        if self.is_valid([pos[0], pos[1] - 1]) and PRE_IDS['agent'] in self._full_obs[pos[0]][pos[1] - 1]:
            neighbours_xy.append([pos[0], pos[1] - 1])
            _count += 1

        agent_id = []
        for x, y in neighbours_xy:
            agent_id.append(int(self._full_obs[x][y].split(PRE_IDS['agent'])[1]) - 1)
        return _count, agent_id

    def __get_neighbour_coordinates(self, pos):
        neighbours = []
        if self.is_valid([pos[0] + 1, pos[1]]):
            neighbours.append([pos[0] + 1, pos[1]])
        if self.is_valid([pos[0] - 1, pos[1]]):
            neighbours.append([pos[0] - 1, pos[1]])
        if self.is_valid([pos[0], pos[1] + 1]):
            neighbours.append([pos[0], pos[1] + 1])
        if self.is_valid([pos[0], pos[1] - 1]):
            neighbours.append([pos[0], pos[1] - 1])
        return neighbours

    def render(self, mode='human'):
        img = copy.copy(self._base_img)


        for agent_i in range(self.n_agents):
            for neighbour in self.__get_neighbour_coordinates(self.agent_pos[agent_i]):
                if self._full_obs[neighbour[0]][neighbour[1]] != PRE_IDS['wall']:
                    fill_cell(img, neighbour, cell_size=CELL_SIZE, fill=AGENT_NEIGHBORHOOD_COLOR, margin=0.1)
                if self._full_obs[neighbour[0]][neighbour[1]] != PRE_IDS['wall']:
                    fill_cell(img, neighbour, cell_size=CELL_SIZE, fill=AGENT_NEIGHBORHOOD_COLOR, margin=0.1)
            fill_cell(img, self.agent_pos[agent_i], cell_size=CELL_SIZE, fill=AGENT_NEIGHBORHOOD_COLOR, margin=0.1)
        
        
        for agent_i in range(self.n_agents):
            draw_circle(img, self.agent_pos[agent_i], cell_size=CELL_SIZE, fill=AGENT_COLOR)
            write_cell_text(img, text=str(agent_i + 1), pos=self.agent_pos[agent_i], cell_size=CELL_SIZE,
                            fill='white', margin=0.4)

        for prey_i in range(self.n_preys):
            if self._prey_alive[prey_i]:
                draw_circle(img, self.prey_pos[prey_i], cell_size=CELL_SIZE, fill=PREY_COLOR)
                write_cell_text(img, text=str(prey_i + 1), pos=self.prey_pos[prey_i], cell_size=CELL_SIZE,
                                fill='white', margin=0.4)

        for prey_i in range(self.n_preys2):
            if self._prey_alive2[prey_i]:
                draw_circle(img, self.prey2_pos[prey_i], cell_size=CELL_SIZE, fill=PREY2_COLOR)
                write_cell_text(img, text=str(prey_i + 1), pos=self.prey2_pos[prey_i], cell_size=CELL_SIZE,
                                fill='white', margin=0.4)

        for prey_i in range(self.n_preys2):
            if self._prey_alive2[prey_i]:
                draw_circle(img, self.prey2_pos[prey_i], cell_size=CELL_SIZE, fill=PREY2_COLOR)
                write_cell_text(img, text=str(prey_i + 1), pos=self.prey2_pos[prey_i], cell_size=CELL_SIZE,
                                fill='white', margin=0.4)

        img = np.asarray(img)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def seed(self, n=None):
        self.np_random, seed = seeding.np_random(n)
        return [seed]

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


AGENT_COLOR = ImageColor.getcolor('blue', mode='RGB')
AGENT_NEIGHBORHOOD_COLOR = (186, 238, 247)
PREY_COLOR = 'red'
PREY2_COLOR = ImageColor.getcolor('green', mode='RGB')
PREY2_COLOR = ImageColor.getcolor('green', mode='RGB')

CELL_SIZE = 35

WALL_COLOR = 'black'

ACTION_MEANING = {
    0: "DOWN",
    1: "LEFT",
    2: "UP",
    3: "RIGHT",
    4: "NOOP",
}

PRE_IDS = {
    'agent': 'A',
    'prey': 'P',
    'prey2': 'P2',
    'wall': 'W',
    'empty': '0'
}