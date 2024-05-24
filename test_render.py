import time
import argparse
import numpy as np

from aasma.wrappers import SingleAgentWrapper
from aasma.tigers_deer import TigersDeer

from agents.random.random_agent import RandomAgent
from agents.random.random_prey import RandomPrey
from agents.random.random_agent import RandomAgent
from agents.random.random_prey import RandomPrey

from agents.greedy.greedy_agent import GreedyAgent
from agents.greedy.greedy_prey import GreedyPrey

from agents.bdi.bdi_agent import BdiAgent
from agents.bdi.bdi_prey import BdiPrey
from agents.greedy.greedy_agent import GreedyAgent
from agents.greedy.greedy_prey import GreedyPrey

from agents.bdi.bdi_agent import BdiAgent
from agents.bdi.bdi_prey import BdiPrey

if __name__ == '__main__':

    """
    Demo for usage of OpenAI Gym's interface for environments.
    --episodes: Number of episodes to run
    --render-sleep-time: Seconds between frames
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--render-sleep-time", type=float, default=0.5)
    opt = parser.parse_args()

    # 1 - Setup the environment
    environment = TigersDeer(
        grid_shape=(12, 12),
        n_agents=4, n_preys=2, n_preys2=2,
        max_steps=100, required_captors=2,
        n_obstacles=1, faster_preys=True,
        higher_vision_preys=True
    )

    # Run
    for episode in range(opt.episodes):
        
        print(f"Episode {episode}")
        
        n_steps = 0

        observations = environment.reset()

        #agents = [GreedyAgent(agent_id=x) for x in range(4)]
        #agents = [RandomAgent(n_actions=5, agent_id=x) for x in range(4)]
        agents = [BdiAgent(agent_id=x, conventions=list(range(4))) for x in range(4)]
        
        #preys = [GreedyPrey(prey_id=x) for x in range(2)]
        #preys = [RandomPrey(n_actions=5, prey_id=x) for x in range(2)]
        preys = [BdiPrey(prey_id=x, family_id=1) for x in range(2)]

        #preys2 = [GreedyPrey(prey_id=x) for x in range(2)]
        #preys2 = [RandomPrey(n_actions=5, prey_id=x) for x in range(2)]
        preys2 = [BdiPrey(prey_id=x, family_id=2) for x in range(2)]

        environment.render()
        time.sleep(opt.render_sleep_time)

        info = {}
        terminals = [False for _ in range(len(agents))]
        info['prey_alive'] = [True for _ in range(len(preys))]
        info['prey_alive2'] = [True for _ in range(len(preys2))]
        while not all(terminals):

            n_steps += 1
            for agent in agents:
                if not (terminals[agent.agent_id]):
                    agent.see(observations[0])
            for prey in preys:
                if info['prey_alive'][prey.prey_id]:
                    prey.see(observations[1])
            for prey2 in preys2:
                prey2.see(observations[2])

            agent_actions, prey_actions, prey2_actions = [], [], []
            for agent in agents:
                if not (terminals[agent.agent_id]):
                    agent_actions.append(agent.run())
                else:
                    agent_actions.append(-1)

            for prey in preys:
                if info['prey_alive'][prey.prey_id]:
                    prey_actions.append(prey.run())
                else:
                    prey_actions.append(-1)

            for prey2 in preys2:
                if info['prey_alive2'][prey2.prey_id]:
                    prey2_actions.append(prey2.run())
                else:
                    prey2_actions.append(-1)

            next_observations, rewards, terminals, info = environment.step(agent_actions, prey_actions, prey2_actions)
            print(f"Timestep {n_steps}")
            environment.render()
            
            time.sleep(opt.render_sleep_time)

            observations = next_observations

        environment.close()