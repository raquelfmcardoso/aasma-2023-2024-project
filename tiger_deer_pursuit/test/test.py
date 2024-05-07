import sys
sys.path.append('/home/raquel/aasma/aasma-2023-2024-project/')  # Add the path to the 'tiger_deer_pursuit' package
from tiger_deer_pursuit import tiger_deer_pursuit_v0

env = tiger_deer_pursuit_v0.env(render_mode="human")
env.reset(seed=42)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        # this is where you would insert your policy
        action = env.action_space(agent).sample()

    env.step(action)
env.close()
