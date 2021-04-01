import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# RANDOM_SEED = 10
# num_agents = 2



def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1.0/np.sqrt(fan_in)
    return (-lim,lim)


class Actor(nn.Module):
    """Actor Model -  Takes observation and produces actions based on the Observation"""
    
    def __init__(self, state_size, action_size,fc1_units = 256, fc2_units = 256):
        """Init Method for initialization of actor model
           =============================================
           Params:
           state_size: Number of Observation
           action_size: Array of actions to be controlled
           fc1_units: Number of hidden units in Fc1 Layer
           fc2_units: Number of hidden units in Fc2 layer
        """
        super(Actor,self).__init__()
#         self.seed = torch.manual_seed(RANDOM_SEED)
        self.fc1 = nn.Linear(state_size ,fc1_units)
        self.fc2 = nn.Linear(fc1_units,fc2_units)
        self.fc3 = nn.Linear(fc2_units,action_size)
        self.reset_parameters()
        
    def reset_parameters(self):
        """Weight Initialization for Actor NN"""
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3,3e-3)
        
    def forward(self,state):
        """Build an actor (policy) network that maps states -> actions."""
        a = F.relu(self.fc1(state))
        a = F.relu(self.fc2(a))
        return F.tanh(self.fc3(a))
    

class Critic(nn.Module):
    """Critic Model -  Takes observation and actions produces policies """
    
    def __init__(self, state_size, action_size,num_agents,fc1_units = 256, fc2_units = 256):
        """Init Method for initialization of critic model
           =============================================
           Params:
           state_size: Number of Observation
           action_size: Array of actions to be controlled
           fc1_units: Number of hidden units in Fc1 Layer
           fc2_units: Number of hidden units in Fc2 layer
        """
        super(Critic,self).__init__()
#         self.seed = torch.manual_seed(RANDOM_SEED)
        self.fc1 = nn.Linear((state_size + action_size) * num_agents,fc1_units)
#         self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units,fc2_units)
        self.fc3 = nn.Linear(fc2_units,1)
        self.bn = nn.BatchNorm1d(fc1_units)
        self.reset_parameters()
        
    def reset_parameters(self):
        """Weight Initialization for Critic NN"""
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3,3e-3)
        
    def forward(self,state,action):
        """Build an critic (policy) network that maps states -> actions."""
        ca = torch.cat((state,action.float()),dim = 1)
        a = F.leaky_relu(self.fc1(ca))
        a = self.bn(a)
        a = F.leaky_relu(self.fc2(a))
        return self.fc3(a)