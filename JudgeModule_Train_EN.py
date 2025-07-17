from JudgeModuleEnv_ML_EN import JudgeModuleEnv
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

current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
writer = SummaryWriter(log_dir=f'JudgeModule_writer/{current_time}')

# Create a directory if it doesn't exist
model_dir = 'JudgeEnvModel'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_path = os.path.join(model_dir, 'model_params_20250323.pth')

# Ensure the directory exists
data_dir = 'JudgeModuleData'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)


def validate_regression_model(model, dataloader, device):
    model.eval()
    mse_sum = 0
    mae_sum = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = model(inputs)

            mse = torch.mean((predictions - targets) ** 2)
            mae = torch.mean(torch.abs(predictions - targets))

            mse_sum += mse.item() * inputs.size(0)  # Multiply by batch size for accurate average
            mae_sum += mae.item() * inputs.size(0)  # Multiply by batch size for accurate average
            total_samples += inputs.size(0)

    avg_mse = mse_sum / total_samples
    avg_mae = mae_sum / total_samples
    avg_rmse = torch.sqrt(torch.tensor(avg_mse))  # Calculate RMSE from MSE

    return avg_mse, avg_rmse.item(), avg_mae


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

            rewards.append(env.energy)

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
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

env = JudgeModuleEnv(GUI_flag=False)

iter_time=500
skill_number=3
# obs,info=env.reset()
observations, actions, rewards = collect_data(env, iter_time)  # Collect 1000 samples
# Save arrays
np.save(os.path.join(data_dir, 'observations20250323.npy'), observations)
np.save(os.path.join(data_dir, 'actions20250323.npy'), actions)
np.save(os.path.join(data_dir, 'rewards20250323.npy'), rewards)

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
validate_dataloader=DataLoader(dataset, batch_size=iter_time*skill_number, shuffle=False)

obs_dim=8
act_dim=1



# Initialize the model
model = RewardPredictor(obs_dim,act_dim).to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("before training")
# Assuming validation_dataloader is properly set up
avg_mse, avg_rmse, avg_mae = validate_regression_model(model, validate_dataloader, device)
print(f"Average MSE: {avg_mse:.2f}")
print(f"Average RMSE: {avg_rmse:.2f}")
print(f"Average MAE: {avg_mae:.2f}")

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
    # avg_loss = total_loss / len(dataloader)
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# Save the model parameters
# torch.save(model.state_dict(), 'model_params.pth')
# Close the TensorBoard writer after training loop
writer.close()
torch.save(model.state_dict(), model_path)

# for classification task
# with torch.no_grad():
#     correct = 0
#     total = len(dataset)
#     for data in dataloader:
#         x, y = data
#         predictions = model(x).round()  # Binary predictions
#         correct += (predictions == y).sum().item()
#     accuracy = correct / total
#     print(f'Accuracy: {accuracy:.2f}')

# Assuming validation_dataloader is properly set up
avg_mse, avg_rmse, avg_mae = validate_regression_model(model, validate_dataloader, device)
print(f"Average MSE: {avg_mse:.2f}")
print(f"Average RMSE: {avg_rmse:.2f}")
print(f"Average MAE: {avg_mae:.2f}")

