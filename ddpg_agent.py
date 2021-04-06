import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

LR_ACTOR = 1e-4         # Learning rate of the actor 
LR_CRITIC = 1e-3        # Learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
TAU = 8e-2              # For soft update of target parameters
GAMMA = 0.99            # Discount Factor


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class OUNoise():
    """Ornstein-Uhlenbeck process"""

    def __init__(self, size, random_seed, mu = 0,theta = 0.15, sigma = 0.1):
        """Initialize parameters and noise process."""
        self.size = size
        self.random_seed = random.seed(random_seed)
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()
        
        
    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)
        
    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx        
        return self.state
    
    
class DDPG_Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, num_agents, index, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            num_agents (int): the number of agents
            index (int): an index for each agent
            random_seed (int): random seed
        """
        
        self.action_size = action_size
        self.state_size = state_size
        self.index = index
        self.random_seed = random.seed(random_seed)
        
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr = LR_ACTOR)
        
        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, num_agents, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, num_agents, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr = LR_CRITIC, weight_decay = WEIGHT_DECAY)
        
        self.hard_update(self.actor_target, self.actor_local)
        self.hard_update(self.critic_target, self.critic_local)
        
        self.noise = OUNoise(action_size, random_seed)
        self.timesteps = 0
    
    def hard_update(self,target,source):
        """Hard update model parameters.
        θ_target = θ_local
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
        """
        for target_params,source_params in zip(target.parameters(),source.parameters()):
            target_params.data.copy_(source_params.data)
        
       
    def act(self, state, add_noise = True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action,-1,1)
 
    
    def reset(self):
        self.noise.reset()
    
    def learn(self,experiences, num_agents):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(next_state) -> action
            critic_target(next_state, next_action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            num_agents (list): number of agents
        """
            
        states, actions, rewards, next_states, dones = experiences
        
        whole_states = torch.cat(states, dim=1).to(device)
        whole_next_states = torch.cat(next_states, dim=1).to(device)
        whole_actions = torch.cat(actions, dim=1).to(device)
        
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions for each agent
        next_actions = [actions[index].clone() for index in range(num_agents)]
        next_actions[self.index] = self.actor_target(next_states[self.index])
        whole_next_actions = torch.cat(next_actions, dim=1).to(device)
        # Get predicted Q values from target models
        Q_target_next = self.critic_target(whole_next_states,whole_next_actions)
        # Compute Q targets for current states (y_i)
        Q_target = rewards[self.index] + GAMMA * Q_target_next *(1-dones[self.index])
        # Compute critic loss
        Q_expected = self.critic_local(whole_states,whole_actions)
        critic_loss = F.mse_loss(Q_expected,Q_target)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()
        
        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss for each agent
        actions_pred = [actions[index].clone() for index in range(num_agents)]
        actions_pred[self.index] = self.actor_local(states[self.index])
        whole_actions_pred = torch.cat(actions_pred, dim=1).to(device)
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss = -self.critic_local(whole_states, whole_actions_pred).mean()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)
            
        
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ * θ_local + (1 - τ) * θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_params,local_params in zip(target_model.parameters(),local_model.parameters()):
            target_params.data.copy_(tau * local_params.data + (1.0 - tau) * target_params.data)
            
