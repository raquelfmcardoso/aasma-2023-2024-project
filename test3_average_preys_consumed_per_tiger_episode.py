import time
import argparse
import numpy as np

from aasma.wrappers import SingleAgentWrapper
from aasma.simplified_predator_prey import SimplifiedPredatorPrey

from aasma.utils import compare_results

from aasma.wrappers import SingleAgentWrapper
from aasma.simplified_predator_prey import SimplifiedPredatorPrey

from agents.random.random_agent import RandomAgent
from agents.random.random_prey import RandomPrey

from agents.greedy.greedy_agent import GreedyAgent
from agents.greedy.greedy_prey import GreedyPrey

from agents.bdi.bdi_agent import BdiAgent
from agents.bdi.bdi_prey import BdiPrey

# change values of the environment and compare images
# change values for -episodes -percentage_greedy and compare images
# change type of agents or preys in TEAMS to see other combinations


def create_agents(agent_type, n_agents, environment, start_id=0):
    if agent_type == "greedy":
        return [GreedyAgent(agent_id=i + start_id) for i in range(n_agents)]
    elif agent_type == "random":
        return [RandomAgent(environment.agent_action_space[i].n, agent_id=i + start_id) for i in range(n_agents)]
    else:
        raise ValueError("Unknown agent type")

def create_preys(prey_type, n_preys, environment, start_id=0):
    if prey_type == "greedy":
        return [GreedyPrey(prey_id=i + start_id) for i in range(n_preys)]
    elif prey_type == "random":
        return [RandomPrey(environment.prey_action_space[i].n, prey_id=i + start_id) for i in range(n_preys)]
    else:
        raise ValueError("Unknown prey type")

def create_mixed_agents(n_agents, percentage_greedy, environment):
    n_greedy = int(n_agents * percentage_greedy)
    n_random = n_agents - n_greedy
    agents = create_agents("greedy", n_greedy, environment)
    agents += create_agents("random", n_random, environment, start_id=len(agents))
    return agents

def create_mixed_preys(n_preys, percentage_greedy, environment):
    n_greedy = int(n_preys * percentage_greedy)
    n_random = n_preys - n_greedy
    preys = create_preys("greedy", n_greedy, environment)
    preys += create_preys("random", n_random, environment, start_id=len(preys))
    return preys


def run_multi_agent(environment, agents, preys, preys2, n_episodes, render_sleep_time):
    results = np.zeros(n_episodes)
    agent_preys_captured = {agent.agent_id: 0 for agent in agents}
    agent_previous_health = {agent.agent_id: 2 for agent in agents}
    # preys_alive_results = np.zeros(n_episodes)
    # preys2_alive_results = np.zeros(n_episodes)

    for episode in range(n_episodes):
        observations = environment.reset()
        info = {}
        info['prey_alive'] = [True for _ in range(len(preys))]
        info['prey_alive2'] = [True for _ in range(len(preys2))]
        terminals = {agent.agent_id: False for agent in agents}

        while not all(terminals):
            for agent in agents:
                if not terminals[agent.agent_id]:
                    agent.see(observations[0])
            for prey in preys:
                if info['prey_alive'][prey.prey_id]:
                    prey.see(observations[1])
            for prey2 in preys2:
                if info['prey_alive2'][prey2.prey_id]:
                    prey2.see(observations[2])

            agent_actions, prey_actions, prey2_actions = [], [], []

            for agent in agents:
                if not terminals[agent.agent_id]:
                    agent_actions.append(agent.action())
                else:
                    agent_actions.append(None)

            for prey in preys:
                if info['prey_alive'][prey.prey_id]:
                    prey_actions.append(prey.action())
                else:
                    prey_actions.append(None)

            for prey2 in preys2:
                if info['prey_alive2'][prey2.prey_id]:
                    prey2_actions.append(prey2.action())
                else:
                    prey2_actions.append(None)

            next_observations, rewards, terminals, info = environment.step(agent_actions, prey_actions, prey2_actions)
            
            environment.render()

            if render_sleep_time > 0:
                time.sleep(render_sleep_time)

            observations = next_observations

            for agent, action in zip(agents, agent_actions):
                if action is not None:
                    if agent_previous_health[agent.agent_id] > 0 and environment._hp_status[agent.agent_id] >= agent_previous_health[agent.agent_id]:
                        agent_preys_captured[agent.agent_id] += 1
                    agent_previous_health[agent.agent_id] = environment._hp_status[agent.agent_id]

        # print(f"agent_captured: {agent_preys_captured}")

        # preys_alive_results[episode] = sum(info['prey_alive'])
        # preys2_alive_results[episode] = sum(info['prey_alive2'])

        #print(f"preys: {preys_alive_results}")
        #print(f"preys2: {preys2_alive_results}")
        
        results[episode] = sum(agent_preys_captured.values()) / environment.n_agents
        #print(results)
    environment.close()

    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--render-sleep-time", type=float, default=0.5)
    parser.add_argument("--percentage-greedy", type=float, default=0.5)
    opt = parser.parse_args()

    environment = SimplifiedPredatorPrey(
        grid_shape=(15, 15),
        n_agents=5, n_preys=5, n_preys2=5,
        max_steps=100, required_captors=1,
        n_obstacles=10
    )

    teams = {
        "Random Team": [
            create_agents("random", environment.n_agents, environment),
            create_preys("random", environment.n_preys, environment),
            create_preys("random", environment.n_preys2, environment)
        ],
        "Greedy Team": [
            create_agents("greedy", environment.n_agents, environment),
            create_preys("greedy", environment.n_preys, environment),
            create_preys("greedy", environment.n_preys2, environment)
        ],
        f"Mixed Team with {opt.percentage_greedy*100}% greedy": [
            create_mixed_agents(environment.n_agents, opt.percentage_greedy, environment),
            create_mixed_preys(environment.n_preys, opt.percentage_greedy, environment),
            create_mixed_preys(environment.n_preys2, opt.percentage_greedy, environment)
        ]
    }

    results = {}

    for team, (agents, preys, preys2) in teams.items():
        result = run_multi_agent(environment, agents, preys, preys2, opt.episodes, opt.render_sleep_time)
        results[team] = result

    filename = f"test3_env{environment._grid_shape}_agents{environment.n_agents}_prey{environment.n_preys}_preytwo{environment.n_preys2}_greedy{opt.percentage_greedy*100:.0f}_episodes{opt.episodes}.png"
    compare_results(
        results, title="Average Preys Consumed per Episode",
        metric="Preys Consumed per Episode",
        colors=["orange", "green", "blue"],
        filename=filename
    )

