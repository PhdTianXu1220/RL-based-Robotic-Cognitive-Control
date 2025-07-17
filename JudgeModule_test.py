import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
from JudgeModule_load import RewardPredictor,action_select



obs=np.array([0.7257,  0.4374,  0.1966, -0.4361,  0.2241, -0.1101, -0.4188,  0.3941])
act=1.0


obs_dim = 8
act_dim = 1

# Setup device
judge_device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Test on {judge_device}")

# Initialize the model
judge_model = RewardPredictor(obs_dim, act_dim).to(judge_device)
judge_model.load_model()
judge_model.eval()

best_action,prediction_list=action_select(obs,judge_model,judge_device)

print(best_action)
print(prediction_list)

# input=np.concatenate((obs,np.array([act])))
# input = torch.tensor(input, dtype=torch.float32).to(device)
#
# predictions = model(input) # Binary predictions
#
# print(predictions.item())