import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
from datetime import datetime
import random
# from JudgeModule_Train import RewardPredictor

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

    def load_model(self):
        # Create a directory if it doesn't exist
        model_dir = 'JudgeEnvModel'
        # model_path = os.path.join(model_dir, 'model_params_20250318.pth')
        model_path = os.path.join(model_dir, 'model_params_20250616.pth')
        self.load_state_dict(torch.load(model_path, weights_only=True))



def action_select(obs,model,device):
    action_list=[1,2,3]
    predictions=[]

    for i in range(len(action_list)):
        act=action_list[i]
        input = np.concatenate((obs, np.array([act])))
        input = torch.tensor(input, dtype=torch.float32).to(device)
        predict= model(input).item()
        predictions.append(predict)

    predictions=np.array(predictions)

    valid_indices = [i for i, val in enumerate(predictions) if val > 0.6]

    if valid_indices:
        best_action = random.choice(valid_indices)+1
        print("bestaction",best_action)
        print(f"Chosen index: {best_action}, value: {predictions[best_action-1]}")
    else:
        best_action=predictions.argmax()+1
    return best_action, predictions











if __name__ == '__main__':
    # Ensure the directory exists
    data_dir = 'JudgeModuleData'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    observations = np.load(os.path.join(data_dir, 'observations20250318.npy'))
    actions = np.load(os.path.join(data_dir, 'actions20250318.npy'))
    rewards = np.load(os.path.join(data_dir, 'rewards20250318.npy'))

    # Setup device
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Test on {device}")

    # Prepare the dataset
    inputs = torch.tensor(np.hstack((observations, actions)), dtype=torch.float32).to(device)
    targets = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    obs_dim = 8
    act_dim = 1

    # Initialize the model
    model = RewardPredictor(obs_dim, act_dim).to(device)
    model.load_model()
    model.eval()

    with torch.no_grad():
        correct = 0
        total = len(dataset)
        for data in dataloader:
            x, y = data
            predictions = model(x).round()  # Binary predictions
            print("skill obs", x,"feasibility",y,"predict",predictions)
            correct += (predictions == y).sum().item()
        accuracy = correct / total
        print(f'Accuracy: {accuracy:.4f}')