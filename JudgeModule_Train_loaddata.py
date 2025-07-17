from JudgeModuleEnv_ML import JudgeModuleEnv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
# Path to the directory containing the package
# package_path = '/home/tianxu/Documents/DMP-python/Dual_Arm_New'
import os
from datetime import datetime

def collect_data(env,iteration_times):
    observations = []
    actions = []
    rewards = []

    for i in range(1,3+1):
        print("skill ID",i)
        for _ in range(iteration_times):
            obs, info = env.reset()
            observations.append(obs)
            actions.append(np.array([i]))
            while True:
                action = 1
                env_end_flag, task_success = env.step(action)
                if env_end_flag == True:
                    print("task_success", task_success)
                    break

            rewards.append(task_success)

    observations = np.array(observations)
    actions = np.array(actions).reshape(-1, 1)  # Ensure actions are 2D
    rewards = np.array(rewards)

    return observations, actions, rewards


class RewardPredictor(nn.Module):
    def __init__(self, obs_size, act_size):
        super(RewardPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size + act_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
writer = SummaryWriter(log_dir=f'JudgeModule_writer/{current_time}')

# Create a directory if it doesn't exist
model_dir = 'JudgeEnvModel'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_path = os.path.join(model_dir, 'model_params_20250616.pth')

# Ensure the directory exists
data_dir = 'JudgeModuleData'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)





# env = JudgeModuleEnv(GUI_flag=False)

# obs,info=env.reset()
# observations, actions, rewards = collect_data(env, 500)  # Collect 1000 samples
# # Save arrays
# np.save(os.path.join(data_dir, 'observations20250318.npy'), observations)
# np.save(os.path.join(data_dir, 'actions20250318.npy'), actions)
# np.save(os.path.join(data_dir, 'rewards20250318.npy'), rewards)

observations = np.load(os.path.join(data_dir, 'observations20250615.npy'))
actions = np.load(os.path.join(data_dir, 'actions20250615.npy'))
rewards = np.load(os.path.join(data_dir, 'rewards20250615.npy'))

print("obs",observations,len(observations))
print("act",actions,len(actions))
print("rewards",rewards,len(rewards))

# Setup device
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}")

# Prepare the dataset
inputs = torch.tensor(np.hstack((observations, actions)), dtype=torch.float32).to(device)
targets = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
dataset = TensorDataset(inputs, targets)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

obs_dim=8
act_dim=1



# Initialize the model
model = RewardPredictor(obs_dim,act_dim).to(device)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    total_loss = 0
    for i, (x, y) in enumerate(dataloader):
        # Send data to the device
        x, y = x.to(device), y.to(device)

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(x)

        # Compute loss
        loss = criterion(output, y)

        # Backpropagation
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Log loss to TensorBoard
        writer.add_scalar('Loss/train', loss.item(), epoch * len(dataloader) + i)

        # Accumulate loss for average calculation
        total_loss += loss.item()

    # Calculate average loss over all batches for the epoch
    avg_loss = total_loss / len(dataloader)

    print(f'Epoch {epoch + 1}, Average Loss: {avg_loss}')

# Save the model parameters
# torch.save(model.state_dict(), 'model_params.pth')
# Close the TensorBoard writer after training loop
writer.close()
torch.save(model.state_dict(), model_path)

with torch.no_grad():
    correct = 0
    total = len(dataset)
    for data in dataloader:
        x, y = data
        predictions = model(x).round()  # Binary predictions
        correct += (predictions == y).sum().item()
    accuracy = correct / total
    print(f'Accuracy: {accuracy:.4f}')
