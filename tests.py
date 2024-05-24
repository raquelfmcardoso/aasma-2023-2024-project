import time
import argparse
import numpy as np
import math
import matplotlib.pyplot as plt

from aasma.wrappers import SingleAgentWrapper
from aasma.simplified_predator_prey import SimplifiedPredatorPrey

from aasma.utils import compare_results

from agents.random.random_agent import RandomAgent
from agents.random.random_prey import RandomPrey

from agents.greedy.greedy_agent import GreedyAgent
from agents.greedy.greedy_prey import GreedyPrey

from agents.bdi.bdi_agent import BdiAgent
from agents.bdi.bdi_prey import BdiPrey


def z_table(confidence):
    """Hand-coded Z-Table

    Parameters
    ----------
    confidence: float
        The confidence level for the z-value.

    Returns
    -------
        The z-value for the confidence level given.
    """
    return {
        0.99: 2.576,
        0.95: 1.96,
        0.90: 1.645
    }[confidence]

def standard_error(std_dev, n, confidence):
    """Computes the standard error of a sample.

    Parameters
    ----------
    std_dev: float
        The standard deviation of the sample
    n: int
        The size of the sample
    confidence: float
        The confidence level for the z-value.

    Returns
    -------
        The standard error.
    """
    return z_table(confidence) * (std_dev / math.sqrt(n))

def plot_confidence_bar(names, means1, std_devs1, N1, y_label1, means2, std_devs2, N2, y_label2, means3, std_devs3, N3, y_label3, means4, std_devs4, N4, y_label4,
                         means5, std_devs5, N5, y_label5, means6, std_devs6, N6, means7, std_devs7, N7, means8, std_devs8, N8,
                         means9, std_devs9, N9, means10, std_devs10, N10, y_label10, means11, std_devs11, N11, means12, std_devs12, N12,
                         means13, std_devs13, N13, means14, std_devs14, N14, title, confidence, show=False, filename=None, colors=None, yscale=None):
    """Creates a bar plot for comparing different agents/teams on two metrics.

    Parameters
    ----------
    names: Sequence[str]
        A sequence of names (representing either the agent names or the team names)
    means1: Sequence[float]
        A sequence of means for the first metric (one mean for each name)
    std_devs1: Sequence[float]
        A sequence of standard deviations for the first metric (one for each name)
    N1: Sequence[int]
        A sequence of sample sizes for the first metric (one for each name)
    y_label1: str
        The label for the y-axis of the first metric
    means2: Sequence[float]
        A sequence of means for the second metric (one mean for each name)
    std_devs2: Sequence[float]
        A sequence of standard deviations for the second metric (one for each name)
    N2: Sequence[int]
        A sequence of sample sizes for the second metric (one for each name)
    y_label2: str
        The label for the y-axis of the second metric
    confidence: float
        The confidence level for the confidence interval
    title: str
        The title of the plot
    show: bool
        Whether to show the plot
    filename: str
        If given, saves the plot to a file
    colors: Optional[Sequence[str]]
        A sequence of colors (one for each name)
    yscale: str
        The scale for the y-axis (default: linear)
    """
    
    errors1 = [standard_error(std_devs1[i], N1[i], confidence) for i in range(len(means1))]
    errors2 = [standard_error(std_devs2[i], N2[i], confidence) for i in range(len(means2))]
    errors3 = [standard_error(std_devs3[i], N3[i], confidence) for i in range(len(means3))]
    errors4 = [standard_error(std_devs4[i], N4[i], confidence) for i in range(len(means4))]
    errors5 = [standard_error(std_devs5[i], N5[i], confidence) for i in range(len(means5))]
    errors6 = standard_error(std_devs6, N6, confidence)
    errors7 = standard_error(std_devs7, N7, confidence)
    errors8 = standard_error(std_devs8, N8, confidence)
    errors9 = standard_error(std_devs9, N9, confidence)
    errors10 = [standard_error(std_devs10[i], N10[i], confidence) for i in range(len(means10))]
    errors11 = standard_error(std_devs11, N11, confidence)
    errors12 = standard_error(std_devs12, N12, confidence)
    errors13 = standard_error(std_devs13, N13, confidence)
    errors14 = standard_error(std_devs14, N14, confidence)
    

    fig, axs = plt.subplots(10, 1, figsize=(18, 90))
    
    x_pos = np.arange(len(names))

    # First metric plot
    axs[0].bar(x_pos, means1, yerr=errors1, align='center', alpha=0.5, color=colors if colors is not None else "red", ecolor='black', capsize=10)
    axs[0].set_ylabel(y_label1)
    axs[0].set_xticks(x_pos)
    axs[0].set_xticklabels(names)
    axs[0].set_title(y_label1)
    axs[0].yaxis.grid(True)
    if yscale is not None:
        axs[0].set_yscale(yscale)

    # Second metric plot
    axs[1].bar(x_pos, means2, yerr=errors2, align='center', alpha=0.5, color=colors if colors is not None else "green", ecolor='black', capsize=10)
    axs[1].set_ylabel(y_label2)
    axs[1].set_xticks(x_pos)
    axs[1].set_xticklabels(names)
    axs[1].set_title(y_label2)
    axs[1].yaxis.grid(True)
    if yscale is not None:
        axs[1].set_yscale(yscale)

    # Third metric plot
    axs[2].bar(x_pos, means3, yerr=errors3, align='center', alpha=0.5, color=colors if colors is not None else "blue", ecolor='black', capsize=10)
    axs[2].set_ylabel(y_label3)
    axs[2].set_xticks(x_pos)
    axs[2].set_xticklabels(names)
    axs[2].set_title(y_label3)
    axs[2].yaxis.grid(True)
    if yscale is not None:
        axs[2].set_yscale(yscale)

    # Fourth metric plot
    axs[3].bar(x_pos, means4, yerr=errors4, align='center', alpha=0.5, color=colors if colors is not None else "grey", ecolor='black', capsize=10)
    axs[3].set_ylabel(y_label4)
    axs[3].set_xticks(x_pos)
    axs[3].set_xticklabels(names)
    axs[3].set_title(y_label4)
    axs[3].yaxis.grid(True)
    if yscale is not None:
        axs[3].set_yscale(yscale)

    # Fifth metric plot
    axs[4].bar(x_pos, means5, yerr=errors5, align='center', alpha=0.5, color=colors if colors is not None else "grey", ecolor='black', capsize=10)
    axs[4].set_ylabel(y_label5)
    axs[4].set_xticks(x_pos)
    axs[4].set_xticklabels(names)
    axs[4].set_title(y_label5)
    axs[4].yaxis.grid(True)
    if yscale is not None:
        axs[4].set_yscale(yscale)

    #family RANDOM WITH GREEDY
    x_pos = np.arange(1)
    # Combined bar plot for preys_greedy_results and preys_random_results
    width = 0.35
    axs[5].bar(x_pos - width/2, means6, yerr=errors6, width=width, align='center', alpha=0.5, color='yellow', ecolor='black', capsize=10, label='Avg number preys greedy alive')
    axs[5].bar(x_pos + width/2, means7, yerr=errors7, width=width, align='center', alpha=0.5, color='orange', ecolor='black', capsize=10, label='Avg number preys random alive')
    axs[5].set_ylabel("Avg. Deer Alive From Family 1")
    axs[5].set_xticks(x_pos)
    axs[5].set_xticklabels(["50% Greedy Deer vs 50% Random Deer"])
    axs[5].set_title("Avg. Number Of Deer Alive From Family 1 Per Episode With 50% Greedy & 50% Random Agents")
    axs[5].yaxis.grid(True)
    axs[5].legend()
    if yscale is not None:
        axs[5].set_yscale(yscale)
    # Combined bar plot for preys2_greedy_results and preys2_random_results 
    width = 0.35
    axs[6].bar(x_pos - width/2, means8, yerr=errors8, width=width, align='center', alpha=0.5, color='yellow', ecolor='black', capsize=10, label='Avg number preys2 greedy alive')
    axs[6].bar(x_pos + width/2, means9, yerr=errors9, width=width, align='center', alpha=0.5, color='orange', ecolor='black', capsize=10, label='Avg number preys2 random alive')
    axs[6].set_ylabel("Avg. Deer Alive From Family 2")
    axs[6].set_xticks(x_pos)
    axs[6].set_xticklabels(["50% Greedy Deer vs 50% Random Deer"])
    axs[6].set_title("Avg. Number Of Deer Alive From Family 2 Per Episode With 50% Greedy & 50% Random Agents")
    axs[6].yaxis.grid(True)
    axs[6].legend()
    if yscale is not None:
        axs[6].set_yscale(yscale)

    # Fifth metric plot
    x_pos = np.arange(len(names))
    axs[7].bar(x_pos, means10, yerr=errors10, align='center', alpha=0.5, color=colors if colors is not None else "purple", ecolor='black', capsize=10)
    axs[7].set_ylabel(y_label10)
    axs[7].set_xticks(x_pos)
    axs[7].set_xticklabels(names)
    axs[7].set_title(y_label10)
    axs[7].yaxis.grid(True)
    if yscale is not None:
        axs[7].set_yscale(yscale)

    #family
    x_pos = np.arange(1)
    # Combined bar plot for preys_greedy_bresults and preys_bdi_results
    width = 0.35
    axs[8].bar(x_pos - width/2, means11, yerr=errors11, width=width, align='center', alpha=0.5, color='purple', ecolor='black', capsize=10, label='Avg number preys greedy alive')
    axs[8].bar(x_pos + width/2, means12, yerr=errors12, width=width, align='center', alpha=0.5, color='pink', ecolor='black', capsize=10, label='Avg number preys bdi alive')
    axs[8].set_ylabel("Avg. Deer Alive From Family 1")
    axs[8].set_xticks(x_pos)
    axs[8].set_xticklabels(["50% Bdi Deer vs 50% Greedy Deer"])
    axs[8].set_title("Avg. Number Of Deer Alive From Family 1 Per Episode With Bdi Tigers vs 50% Bdi & 50% Greedy Deer")
    axs[8].yaxis.grid(True)
    axs[8].legend()
    if yscale is not None:
        axs[8].set_yscale(yscale)

    # Combined bar plot for preys2_greedy_bresults and preys2_bdi_results
    width = 0.35
    axs[9].bar(x_pos - width/2, means13, yerr=errors13, width=width, align='center', alpha=0.5, color='purple', ecolor='black', capsize=10, label='Avg number preys2 greedy alive')
    axs[9].bar(x_pos + width/2, means14, yerr=errors14, width=width, align='center', alpha=0.5, color='pink', ecolor='black', capsize=10, label='Avg number preys2 bdi alive')
    axs[9].set_ylabel("Avg. Deer Alive From Family 2")
    axs[9].set_xticks(x_pos)
    axs[9].set_xticklabels(["50% Bdi Deer vs 50% Greedy Deer"])
    axs[9].set_title("Avg. Number Of Deer Alive From Family 2 Per Episode With Bdi Tigers vs 50% Bdi & 50% Greedy Deer")
    axs[9].yaxis.grid(True)
    axs[9].legend()
    if yscale is not None:
        axs[9].set_yscale(yscale)


    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    if show:
        plt.show()
    plt.close()

def compare_results(results1, results2, results3, results4, results5, results6, results7, results8, results9, results10, results11, results12, 
                    results13, results14, confidence=0.95, title=None, metric1="Steps To Capture Deer Per Episode", metric2="Deer Captured Per Episode", metric3="Steps Per Episode", metric4="Number Of Deer From Family 1 Alive", metric5="Number Of Deer From Family 2 Alive", metric10= "Success Rate Per Episode", colors=None, filename=None):
    """Displays a combined bar plot comparing the performance of different agents/teams on two metrics.
    Parameters
    ----------
    results1: dict
        A dictionary where keys are the names and the values are sequences of trials for the first metric
    results2: dict
        A dictionary where keys are the names and the values are sequences of trials for the second metric
    confidence: float
        The confidence level for the confidence interval
    title: str
        The title of the plot
    metric1: str
        The name of the first metric for comparison
    metric2: str
        The name of the second metric for comparison
    colors: Sequence[str]
        A sequence of colors (one for each agent/team)
    """

    #steps to capture
    names = list(results1.keys())
    means1 = [result.mean() for result in results1.values()]
    stds1 = [result.std() for result in results1.values()]
    N1 = [result.size for result in results1.values()]

    #preys captured
    means2 = [result.mean() for result in results2.values()]
    stds2 = [result.std() for result in results2.values()]
    N2 = [result.size for result in results2.values()]

    #steps
    means3 = [result.mean() for result in results3.values()]
    stds3 = [result.std() for result in results3.values()]
    N3 = [result.size for result in results3.values()]

    #prey1 alive
    means4 = [result.mean() for result in results4.values()]
    stds4 = [result.std() for result in results4.values()]
    N4 = [result.size for result in results4.values()]

    #prey2 alive
    means5 = [result.mean() for result in results5.values()]
    stds5 = [result.std() for result in results5.values()]
    N5 = [result.size for result in results5.values()]

    # preys_greedy_results, preys_random_results, preys2_greedy_results, preys2_random_results
    #prey greedy random
    means6 = np.mean(results6)
    stds6 = np.std(results6)
    N6 = len(results6)
    means7 = np.mean(results7)
    stds7 = np.std(results7)
    N7 = len(results7)
    #prey2 greedy random
    means8 = np.mean(results8)
    stds8 = np.std(results8)
    N8 = len(results8)
    means9 = np.mean(results9)
    stds9 = np.std(results9)
    N9 = len(results9)

    #success rate
    means10 = [result.mean() for result in results10.values()]
    stds10 = [result.std() for result in results10.values()]
    N10 = [result.size for result in results10.values()]
    
    # preys_greedy_bresults, preys_bdi_results, preys2_greedy_bresults, preys2_bdi_results
    #prey greedy bdi
    means11 = np.mean(results11)
    stds11 = np.std(results11)
    N11 = len(results11)
    means12 = np.mean(results12)
    stds12 = np.std(results12)
    N12= len(results12)

    #prey2 greedy bdi
    means13 = np.mean(results13)
    stds13 = np.std(results13)
    N13 = len(results13)
    means14 = np.mean(results14)
    stds14 = np.std(results14)
    N14 = len(results14)

    plot_confidence_bar(
        names=names,
        means1=means1,
        std_devs1=stds1,
        N1=N1,
        y_label1=f"Avg. {metric1}",
        means2=means2,
        std_devs2=stds2,
        N2=N2,
        y_label2=f"Avg. {metric2}",
        means3=means3,
        std_devs3=stds3,
        N3=N3,
        y_label3=f"Avg. {metric3}",
        means4=means4,
        std_devs4=stds4,
        N4=N4,
        y_label4=f"Avg. {metric4}",
        means5=means5,
        std_devs5=stds5,
        N5=N5,
        y_label5=f"Avg. {metric5}",
        means6=means6,
        std_devs6=stds6,
        N6=N6,
        means7=means7,
        std_devs7=stds7,
        N7=N7,
        means8=means8,
        std_devs8=stds8,
        N8=N8,
        means9=means9,
        std_devs9=stds9,
        N9=N9,
        means10=means10,
        std_devs10=stds10,
        N10=N10,
        y_label10=f"Avg. {metric10}",
        means11=means11,
        std_devs11=stds11,
        N11=N11,
        means12=means12,
        std_devs12=stds12,
        N12=N12,
        means13=means13,
        std_devs13=stds13,
        N13=N13,
        means14=means14,
        std_devs14=stds14,
        N14=N14,

        confidence=confidence,
        title=title,
        show=True,
        colors=colors,
        filename=filename

    )

def get_prey_behavior(prey):
    if isinstance(prey, GreedyPrey):
        return "greedy"
    elif isinstance(prey, RandomPrey):
        return "random"
    elif isinstance(prey, BdiPrey):
        return "bdi"
    else:
        return "unknown"

def create_agents(agent_type, n_agents, environment, start_id=0):
    if agent_type == "greedy":
        return [GreedyAgent(agent_id=i + start_id) for i in range(n_agents)]
    elif agent_type == "random":
        return [RandomAgent(environment.agent_action_space[i].n, agent_id=i + start_id) for i in range(n_agents)]
    elif agent_type == "bdi":
        return [BdiAgent(agent_id=i + start_id, conventions=list(range(n_agents))) for i in range(n_agents)]
    else:
        raise ValueError("Unknown agent type")

def create_preys(prey_type, n_preys, environment, start_id=0):
    if prey_type == "greedy":
        return [GreedyPrey(prey_id=i + start_id) for i in range(n_preys)]
    elif prey_type == "random":
        return [RandomPrey(environment.prey_action_space[i].n, prey_id=i + start_id) for i in range(n_preys)]
    else:
        raise ValueError("Unknown prey type")
    
def create_bdi_preys(n_preys, family_id, start_id=0):
    return [BdiPrey(prey_id=i + start_id, family_id=family_id) for i in range(n_preys)]

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

def create_mixed_preys_bdi(n_preys, percentage_bdi, environment, family_id):
    n_bdi = int(n_preys * percentage_bdi)
    n_greedy = n_preys - n_bdi
    preys = create_bdi_preys(n_bdi, family_id)
    preys += create_preys("greedy", n_greedy, environment, start_id=len(preys))
    return preys

def run_multi_agent(environments, agents, preys, preys2, n_episodes, render_sleep_time, max_steps, required_captors):
    steps_results = np.zeros(n_episodes)
    preys_captured_results = np.zeros(n_episodes)
    results = np.zeros(n_episodes)
    preys_alive_results = np.zeros(n_episodes)
    preys2_alive_results = np.zeros(n_episodes)
    success_rate = np.zeros(n_episodes)
    agent_steps = {agent.agent_id: [] for agent in agents}
    agent_preys_captured = {agent.agent_id: 0 for agent in agents}
    agent_previous_health = {agent.agent_id: 2 for agent in agents}
    preys_alive_info = [[] for _ in range(n_episodes)] 
    preys2_alive_info = [[] for _ in range(n_episodes)] 

    for episode in range(n_episodes):
        print(f"Episode: {episode}")
        environment = environments[episode]
        environment.required_captors = required_captors
        steps = 0
        observations = environment.reset()
        info = {}
        info['prey_alive'] = [True for _ in range(len(preys))]
        info['prey_alive2'] = [True for _ in range(len(preys2))]
        terminals = {agent.agent_id: False for agent in agents}

        while not all(terminals):
            steps += 1
            if steps >= max_steps:
                break  # Exit the loop if maximum steps are reached
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

            # print(f"Terminals in tests: {terminals}\nInfo of preys in tests: {info['prey_alive']}\nInfo of preys2 in tests: {info['prey_alive2']}")
            # print(f"Agent ids in tests: {[x.agent_id for x in agents]}")
            # print(f"Preys ids in tests: {[x.prey_id for x in preys]}")
            # print(f"Preys2 ids in tests: {[x.prey_id for x in preys2]}")
            for agent in agents:
                if not terminals[agent.agent_id]:
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
            #print(f"Agent action in tests: {agent_actions}\nPrey actions in tests: {prey_actions}\nPrey2 actions in tests: {prey2_actions}")

            next_observations, rewards, terminals, info = environment.step(agent_actions, prey_actions, prey2_actions)
            
            #environment.render()
            if render_sleep_time > 0:
                time.sleep(render_sleep_time)

            observations = next_observations

            #check preys captured per agent
            for agent, action in zip(agents, agent_actions):
                if action is not None:
                    if agent_previous_health[agent.agent_id] > 0 and environment._hp_status[agent.agent_id] >= agent_previous_health[agent.agent_id]:
                        agent_preys_captured[agent.agent_id] += 1
                        agent_steps[agent.agent_id].append(steps)
                    agent_previous_health[agent.agent_id] = environment._hp_status[agent.agent_id]


        #calculate average steps per agent
        average_steps_per_agent = []
        for agent_id, steps_list in agent_steps.items():
            if agent_preys_captured[agent_id] > 0:
                average_steps = sum(steps_list) / agent_preys_captured[agent_id]
                average_steps_per_agent.append(average_steps)
            else:
                average_steps_per_agent.append(0)

        # Calculates the average steps per agent to consume for this episode
        steps_results[episode] = sum(average_steps_per_agent) / environment.n_agents
        # And the average preys consumed per agent for this episode
        preys_captured_results[episode] = sum(agent_preys_captured.values()) / environment.n_agents
        # And the steps for this episode
        results[episode] = steps
        # And the preys alive for this episode
        preys_alive_results[episode] = sum(info['prey_alive'])
        # And the preys2 alive for this episode
        preys2_alive_results[episode] = sum(info['prey_alive2'])

        # Passing the preys info per episode to check if random or greedy
        preys_alive_info[episode] = [prey for prey, alive in zip(preys, info['prey_alive']) if alive]
        preys2_alive_info[episode] = [prey for prey, alive in zip(preys2, info['prey_alive2']) if alive]

        # Success rate
        if steps >= max_steps:
            success_rate[episode] = 0
        else:
            prey_alive = sum(info['prey_alive'])
            prey2_alive = sum(info['prey_alive2'])
            success_rate[episode] = (environment.n_preys - prey_alive)/environment.n_preys if prey_alive < prey2_alive \
                else (environment.n_preys2 - prey2_alive)/environment.n_preys2

    environment.close()

    return steps_results, preys_captured_results, results, preys_alive_results, preys2_alive_results, preys_alive_info, preys2_alive_info, success_rate


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--render-sleep-time", type=float, default=0.5)
    parser.add_argument("--percentage-greedy", type=float, default=0.5)
    parser.add_argument("--percentage-bdi", type=float, default=0.5)
    opt = parser.parse_args()

    environments = []
    n_agents = 10
    n_preys = 6
    n_preys2 = 6

    for i in range(opt.episodes):
        environments.append(
            SimplifiedPredatorPrey(
                grid_shape=(15, 15),
                n_agents=10, n_preys=6, n_preys2=6,
                max_steps=100, required_captors=1,
                n_obstacles=12
            )
        )

    teams = {
        "All Random": [
            create_agents("random", environments[0].n_agents, environments[0]),
            create_preys("random", environments[0].n_preys, environments[0]),
            create_preys("random", environments[0].n_preys2, environments[0])
        ],
        "All Greedy": [
            create_agents("greedy", environments[0].n_agents, environments[0]),
            create_preys("greedy", environments[0].n_preys, environments[0]),
            create_preys("greedy", environments[0].n_preys2, environments[0])
        ],
        "All Bdi": [
            create_agents("bdi", environments[0].n_agents, environments[0]),
            create_bdi_preys(environments[0].n_preys, family_id=1),
            create_bdi_preys(environments[0].n_preys2, family_id=2)
        ],
        f"{opt.percentage_greedy*100}% Greedy & {(1 - opt.percentage_greedy)*100}% Random Tigers vs \n{opt.percentage_greedy*100}% Greedy & {(1 - opt.percentage_greedy)*100}% Random Deer": [
            create_mixed_agents(environments[0].n_agents, opt.percentage_greedy, environments[0]),
            create_mixed_preys(environments[0].n_preys, opt.percentage_greedy, environments[0]),
            create_mixed_preys(environments[0].n_preys2, opt.percentage_greedy, environments[0])
        ],
        f"Bdi Tigers vs {opt.percentage_bdi*100}% Bdi & {(1 - opt.percentage_bdi)*100}% Greedy Deer": [
            create_agents("bdi", environments[0].n_agents, environments[0]),
            create_mixed_preys_bdi(environments[0].n_preys, opt.percentage_bdi, environments[0], family_id=1),
            create_mixed_preys_bdi(environments[0].n_preys2, opt.percentage_bdi, environments[0], family_id=2)
        ]

    }

    steps_results = {}
    preys_captured_results = {}
    results = {}
    preys_alive_results = {}
    preys2_alive_results = {}
    preys_greedy_results = []
    preys_greedy_bresults = []
    preys_random_results = []
    preys2_greedy_results = []
    preys2_greedy_bresults = []
    preys2_random_results = []
    preys_bdi_results = []
    preys2_bdi_results = []
    success_rate ={}

    for team, (agents, preys, preys2) in teams.items():
        print(f"Team: {team}")
        steps_result, preys_captured_result, result, preys_alive, preys2_alive, preys_alive_info, preys2_alive_info, success = \
            run_multi_agent(environments, agents, preys, preys2, opt.episodes, opt.render_sleep_time, environments[0]._max_steps, 2 if "Bdi" in team else 1)
        steps_results[team] = steps_result
        preys_captured_results[team] = preys_captured_result
        results[team] = result
        preys_alive_results[team] = preys_alive
        preys2_alive_results[team] = preys2_alive
        success_rate[team] = success

        if "Random Tigers vs" in team:
            
            for episode in range(opt.episodes):
                preys_greedy_count = sum(1 for prey, behavior in zip(preys, map(get_prey_behavior, preys_alive_info[episode])) if behavior == "greedy")
                preys_random_count = sum(1 for prey, behavior in zip(preys, map(get_prey_behavior, preys_alive_info[episode])) if behavior == "random")                
                preys_greedy_results.append(preys_greedy_count)
                preys_random_results.append(preys_random_count)

                preys2_greedy_count = sum(1 for prey, behavior in zip(preys2, map(get_prey_behavior, preys2_alive_info[episode])) if behavior == "greedy")
                preys2_random_count = sum(1 for prey, behavior in zip(preys2, map(get_prey_behavior, preys2_alive_info[episode])) if behavior == "random")
                preys2_greedy_results.append(preys2_greedy_count)
                preys2_random_results.append(preys2_random_count)
            
            # print([preys_greedy_results])
            # print([preys_random_results])
            # print([preys2_greedy_results])
            # print([preys2_random_results])  
        
        if "Bdi Tigers vs" in team:
            
            for episode in range(opt.episodes):
                preys_greedy_bcount = sum(1 for prey, behavior in zip(preys, map(get_prey_behavior, preys_alive_info[episode])) if behavior == "greedy")
                preys_bdi_count = sum(1 for prey, behavior in zip(preys, map(get_prey_behavior, preys_alive_info[episode])) if behavior == "bdi")
                
                preys_greedy_bresults.append(preys_greedy_bcount)
                preys_bdi_results.append(preys_bdi_count)

                preys2_greedy_bcount = sum(1 for prey, behavior in zip(preys2, map(get_prey_behavior, preys2_alive_info[episode])) if behavior == "greedy")
                preys2_bdi_count = sum(1 for prey, behavior in zip(preys2, map(get_prey_behavior, preys2_alive_info[episode])) if behavior == "bdi")
                preys2_greedy_bresults.append(preys2_greedy_bcount)
                preys2_bdi_results.append(preys2_bdi_count)

    #print(success_rate)
    #Sprint(steps_results)

    filename = f"images/results_env{environments[0]._grid_shape}_agents{environments[0].n_agents}_prey{environments[0].n_preys}_preytwo{environments[0].n_preys2}_greedy{opt.percentage_greedy*100:.0f}_episodes{opt.episodes}.png"

    compare_results(
        steps_results, preys_captured_results, results,preys_alive_results, preys2_alive_results,
        preys_greedy_results, preys_random_results, preys2_greedy_results, preys2_random_results, success_rate,
        preys_greedy_bresults, preys_bdi_results, preys2_greedy_bresults, preys2_bdi_results,
        filename=filename
    )

