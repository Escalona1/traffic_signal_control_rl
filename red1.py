import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DQNnetwork(nn.Module):
    def __init__(self,dim_input, dim_output):
        super(DQNnetwork,self).__init__()
        self.fc1 = nn.Linear(dim_input,256)
        self.fc2 = nn.Linear(256,256)
        self.output = nn.Linear(256,dim_output)

    def forward(self,x):
        x_mod = F.relu(self.fc1(x))
        x_mod = F.relu(self.fc2(x_mod))
        x_mod = self.output(x_mod)
        
        return x_mod
        