import gymnasium
import gymnasium_env
import time
import numpy as np
from gymnasium.wrappers import FlattenObservation

# Create the environment with render mode
env = gymnasium.make('gymnasium_env/KnightWorldEnv-v0', size=12, render_mode="human")
wrapped_env = FlattenObservation(env)
env = wrapped_env

# Load the Q-table and parameters
Q = np.load("Q_table.npy")
grid_size = int(np.load("grid_size.npy")[0])

# Use the same state mapping function as in training
def state_to_index(state):
    # For a 12x12 grid, we need to encode agent and target positions
    agent_pos = state[0] * grid_size + state[1]  # agent y*width + x
    target_pos = state[2] * grid_size + state[3]  # target y*width + x
    # Create a unique index based on both positions
    return int(agent_pos * (grid_size * grid_size) + target_pos)

obs, _ = env.reset()
state_idx = state_to_index(obs)
done = False
total_reward = 0
max_steps = 100

print("\nWytrenowany agent w akcji:")

for _ in range(max_steps):
    time.sleep(0.2)
    action = np.argmax(Q[state_idx])
    next_obs, reward, terminated, truncated, _ = env.step(action)
    next_state_idx = state_to_index(next_obs)
    done = terminated or truncated

    total_reward += reward
    state_idx = next_state_idx

    if done:
        break

print(f"\nSuma nagród zdobytych przez agenta: {total_reward}")
env.close()