import time
import argparse
import numpy as np

from aasma.wrappers import SingleAgentWrapper
from aasma.simplified_predator_prey import SimplifiedPredatorPrey

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
    environment = SimplifiedPredatorPrey(
        grid_shape=(12, 12),
        n_agents=4, n_preys=2, n_preys2=2,
        max_steps=100, required_captors=2,
        n_obstacles=1, faster_preys=False,
        higher_vision_preys=False
    )

    # Run
    for episode in range(opt.episodes):
        
        print(f"Episode {episode}")
        
        n_steps = 0

        observations = environment.reset()

        agents = [GreedyAgent(agent_id=x) for x in range(2)]
        #agents = [RandomAgent(n_actions=5, agent_id=x) for x in range(2)]
        agents = [BdiAgent(agent_id=x, conventions=list(range(4))) for x in range(4)]
        
        preys = [GreedyPrey(prey_id=x) for x in range(2)]
        preys = [RandomPrey(n_actions=5, prey_id=x) for x in range(1)]

        preys2 = [GreedyPrey(prey_id=x) for x in range(2)]
        preys2 = [RandomPrey(n_actions=5, prey_id=x) for x in range(1)]

        environment.render()
        time.sleep(opt.render_sleep_time)

        time.sleep(10)

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
                #print(agent_actions)

            for prey in preys:
                if info['prey_alive'][prey.prey_id]:
                    prey_actions.append(prey.action())
                else:
                    prey_actions.append(-1)
                #print(prey_actions)

            for prey2 in preys2:
                if info['prey_alive2'][prey2.prey_id]:
                    prey2_actions.append(prey2.action())
                else:
                    prey2_actions.append(-1)
                #print(prey2_actions)
            time.sleep(10)

            #agent_actions = [agent.run() for agent in agents]
            #prey_actions = [prey.run() for prey in preys]
            #prey2_actions = [prey2.run() for prey2 in preys2]
            next_observations, rewards, terminals, info = environment.step(agent_actions, prey_actions, prey2_actions)
            print(f"Timestep {n_steps}")
            # print(f"\t Agent Observations: {observations[0]}")
            # print(f"\t Prey Observations: {observations[1]}")
            # print(f"\t Prey2 Observations: {observations[2]}")
            # for n_action, action in enumerate(agent_actions):
            #     print(f"\tAgent {n_action} Action: {action}\n")
            # for n_action, action in enumerate(prey_actions):
            #     print(f"\tPrey {n_action} Action: {action}\n")
            # for n_action, action in enumerate(prey2_actions):
            #     print(f"\tPrey2 {n_action} Action: {action}\n")
            # print(f"\t Agent Observations: {next_observations[0]}")
            # print(f"\t Prey Observations: {next_observations[1]}")
            # print(f"\t Prey2 Observations: {next_observations[2]}")
            # for action in agent_actions:
            #     print(f"\tAgent Action: {action}\n")
            # for action in prey_actions:
            #     print(f"\tPrey Action: {action}\n")
            # for action in prey2_actions:
            #     print(f"\tPrey2 Action: {action}\n")
            environment.render()
            
            time.sleep(opt.render_sleep_time)

            observations = next_observations

        environment.close()