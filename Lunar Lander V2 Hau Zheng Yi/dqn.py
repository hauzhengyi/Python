# DEEP Q-LEARNING NETWORK CLASS

import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, num_states_available): 
        super().__init__()
        self.fc1 = nn.Linear(in_features=num_states_available, out_features=100)   
        self.fc2 = nn.Linear(in_features=100, out_features=100)
        self.out = nn.Linear(in_features=100, out_features=4)
        
    def forward(self, t):
        t = t.flatten(start_dim=1)
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t