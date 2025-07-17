import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def load_data(path):
    with open(path) as f:
        data = json.load(f)
    steps = [entry[1] for entry in data]
    rewards = [entry[2] for entry in data]
    return steps, rewards

def moving_average(data, window_size=300):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def filter_by_x_limit(x, y, x_max=300000):
    x_filtered = []
    y_filtered = []
    for xi, yi in zip(x, y):
        if xi <= x_max:
            x_filtered.append(xi)
            y_filtered.append(yi)
        else:
            break
    return x_filtered, y_filtered

# Load data for each method
steps_ppo, rewards_ppo = load_data('PPO 2025-06-10 16_58.json')
steps_ddpg, rewards_ddpg = load_data('DDPG 2025-06-12 17_59.json')
steps_sac, rewards_sac = load_data('SAC 2025-06-12 18_36.json')

# Smooth rewards
window = 10
rewards_ppo_smooth = moving_average(rewards_ppo, window)
steps_ppo_smooth = steps_ppo[:len(rewards_ppo_smooth)]

rewards_ddpg_smooth = moving_average(rewards_ddpg, window)
steps_ddpg_smooth = steps_ddpg[:len(rewards_ddpg_smooth)]

rewards_sac_smooth = moving_average(rewards_sac, window)
steps_sac_smooth = steps_sac[:len(rewards_sac_smooth)]

# Apply filtering to all three methods
steps_ppo_smooth, rewards_ppo_smooth = filter_by_x_limit(steps_ppo_smooth, rewards_ppo_smooth)
steps_ddpg_smooth, rewards_ddpg_smooth = filter_by_x_limit(steps_ddpg_smooth, rewards_ddpg_smooth)
steps_sac_smooth, rewards_sac_smooth = filter_by_x_limit(steps_sac_smooth, rewards_sac_smooth)

plt.rcParams['font.family'] = 'Nimbus Roman'
plt.rcParams['font.size'] = 20
plt.figure(figsize=(10, 6))  # Set the size of the plot
# Plot
plt.plot(steps_ppo_smooth, rewards_ppo_smooth, label='PPO',linewidth=3.5,color="red")
plt.plot(steps_ddpg_smooth, rewards_ddpg_smooth, label='DDPG',linewidth=3.5,color="blue")
plt.plot(steps_sac_smooth, rewards_sac_smooth, label='SAC',linewidth=3.5,color="green")

# Labels and formatting
plt.xlabel('Episode')
plt.ylabel('Episode Reward')
plt.title('Training Curves')
# plt.legend(loc='upper left')
plt.legend(loc='center right')
plt.grid(True)
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x/1000)}k'))


plt.savefig('Training Result.png', dpi=600, format='png')
plt.show()