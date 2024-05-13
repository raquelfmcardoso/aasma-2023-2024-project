import time
import argparse
import numpy as np

from aasma.wrappers import SingleAgentWrapper
from aasma.simplified_predator_prey import SimplifiedPredatorPrey

from exercise_1_single_random_agent import RandomAgent
from exercise_1_single_random_agent import RandomPrey

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
        grid_shape=(17, 15),
        n_agents=4, n_preys=1,
        max_steps=100, required_captors=1
    )

    # Run
    for episode in range(opt.episodes):
        
        print(f"Episode {episode}")
        
        n_steps = 0

        observations = environment.reset()

        agents = [RandomAgent(environment.agent_action_space[0].n),
                  RandomAgent(environment.agent_action_space[1].n),
                  RandomAgent(environment.agent_action_space[2].n),
                  RandomAgent(environment.agent_action_space[3].n)]
        
        preys = [RandomPrey(environment.prey_action_space[0].n)]

        environment.render()
        time.sleep(opt.render_sleep_time)

        terminals = [False for _ in range(len(agents))]
        while not all(terminals):
            
            n_steps += 1
            for observations, agent, prey in zip(observations, agents, preys):
                agent.see(observations)
                prey.see(observations)
            agent_actions = [agent.action() for agent in agents]
            prey_actions = [prey.action() for prey in preys]
            next_observations, rewards, terminals, info = environment.step(agent_actions, prey_actions)

            print(f"Timestep {n_steps}")
            print(f"\tObservation: {observations}")
            for action in agent_actions:
                print(f"\tAgent Action: {action}\n")
            for action in prey_actions:
                print(f"\tPrey Action: {action}\n")
            environment.render()
            time.sleep(opt.render_sleep_time)

            observations = next_observations

        environment.close()
