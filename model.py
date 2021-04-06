import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1.0/np.sqrt(fan_in)
    return (-lim,lim)


class Actor(nn.Module):
    """Actor (Policy) Model."""
    
    def __init__(self, state_size, action_size, random_seed, fc1_units = 512, fc2_units = 256):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            random_seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor,self).__init__()
        self.random_seed = torch.manual_seed(random_seed)
        
        self.fc1 = nn.Linear(state_size ,fc1_units)
        self.fc2 = nn.Linear(fc1_units,fc2_units)
        self.fc3 = nn.Linear(fc2_units,action_size)
        self.reset_parameters()
        
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3,3e-3)
        
    def forward(self,state):
        """Build an actor (policy) network that maps states -> actions."""
        a = F.relu(self.fc1(state))
        a = F.relu(self.fc2(a))
        return F.tanh(self.fc3(a))
    

class Critic(nn.Module):
    """Critic (Value) Model."""    
    
    def __init__(self, state_size, action_size,num_agents, random_seed, fc1_units = 512, fc2_units = 256):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            random_seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic,self).__init__()
        self.random_seed = torch.manual_seed(random_seed)
        
        self.fc1 = nn.Linear((state_size + action_size) * num_agents,fc1_units)
        self.fc2 = nn.Linear(fc1_units,fc2_units)
        self.fc3 = nn.Linear(fc2_units,1)
        self.bn = nn.BatchNorm1d(fc1_units)
        self.reset_parameters()
        
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3,3e-3)
        
    def forward(self,state,action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        ca = torch.cat((state,action.float()),dim = 1)
        a = F.leaky_relu(self.fc1(ca))
        a = self.bn(a)
        a = F.leaky_relu(self.fc2(a))
        return self.fc3(a)