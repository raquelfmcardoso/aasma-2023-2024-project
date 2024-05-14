import time
import argparse
import numpy as np

from aasma.wrappers import SingleAgentWrapper
from aasma.simplified_predator_prey import SimplifiedPredatorPrey

from exercise_1_single_random_agent import RandomAgent
from exercise_1_single_random_agent import RandomPrey

from exercise_2_single_random_vs_greedy import GreedyAgent

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
        grid_shape=(30, 30),
        n_agents=10, n_preys=4, n_preys2=4,
        max_steps=100, required_captors=1,
        n_obstacles=20
    )

    # Run
    for episode in range(opt.episodes):
        
        print(f"Episode {episode}")
        
        n_steps = 0

        observations = environment.reset()

        agents = [GreedyAgent(agent_id=0, n_agents=10),
                  GreedyAgent(agent_id=1, n_agents=10),
                  GreedyAgent(agent_id=2, n_agents=10),
                  GreedyAgent(agent_id=3, n_agents=10),
                  GreedyAgent(agent_id=4, n_agents=10),
                  GreedyAgent(agent_id=5, n_agents=10),
                  GreedyAgent(agent_id=6, n_agents=10),
                  GreedyAgent(agent_id=7, n_agents=10),
                  GreedyAgent(agent_id=8, n_agents=10),
                  GreedyAgent(agent_id=9, n_agents=10)
                  ]
        # agents = [RandomAgent(environment.prey_action_space[0].n),
        #          RandomAgent(environment.prey_action_space[1].n),
        #          RandomAgent(environment.prey_action_space[2].n),
        #          RandomAgent(environment.prey_action_space[3].n),
        #          RandomAgent(environment.prey_action_space[4].n),
        #          RandomAgent(environment.prey_action_space[5].n),
        #          RandomAgent(environment.prey_action_space[6].n),
        #          RandomAgent(environment.prey_action_space[7].n),
        #          RandomAgent(environment.prey_action_space[8].n),
        #          RandomAgent(environment.prey_action_space[9].n)
        #          ]
        
        preys = [RandomPrey(environment.prey_action_space[0].n),
                 RandomPrey(environment.prey_action_space[1].n),
                 RandomPrey(environment.prey_action_space[2].n),
                 RandomPrey(environment.prey_action_space[3].n)
                 ]

        preys2 = [RandomPrey(environment.prey2_action_space[0].n), 
                  RandomPrey(environment.prey2_action_space[1].n),
                  RandomPrey(environment.prey2_action_space[2].n),
                  RandomPrey(environment.prey2_action_space[3].n)
                  ]

        environment.render()
        time.sleep(opt.render_sleep_time)

        terminals = [False for _ in range(len(agents))]
        print("Obs:", observations)
        while not all(terminals):
            
            n_steps += 1
            for observations, agent, prey, prey2 in zip(observations, agents, preys, preys2):
                agent.see(observations)
                prey.see(observations)
                prey2.see(observations)
            agent_actions = [agent.action() for agent in agents]
            prey_actions = [prey.action() for prey in preys]
            prey2_actions = [prey2.action() for prey2 in preys2]
            next_observations, rewards, terminals, info = environment.step(agent_actions, prey_actions, prey2_actions)

            print(f"Timestep {n_steps}")
            print(f"\tObservation: {observations}")
            for action in agent_actions:
                print(f"\tAgent Action: {action}\n")
            for action in prey_actions:
                print(f"\tPrey Action: {action}\n")
            for action in prey2_actions:
                print(f"\tPrey2 Action: {action}\n")
            environment.render()
            time.sleep(opt.render_sleep_time)

            observations = next_observations

        environment.close()