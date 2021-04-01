from collections import namedtuple, deque
import random
import numpy as np
import torch

# import copy

from ddpg_agent import DDPG_Agent

import torch.nn.functional as F
import torch.optim as optim
import torch.cuda

# BUFFER_SIZE = int(1e5)  # replay buffer size
# BATCH_SIZE = 256        # minibatch size
# GAMMA = 0.99            # discount factor
# TAU = 1e-3              # for soft update of target parameters
# UPDATE_EVERY = 2        # Udpate every
# NB_LEARN = 3

BUFFER_SIZE = int(1e6)          # Replay buffer size
BATCH_SIZE = 512                # Batch size for training models
# num_agents = 2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer():
    def __init__(self,buffer_size,batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.batch_size = batch_size
        self.memory = deque(maxlen = buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",field_names= ["states","actions","rewards","next_states","dones"])
        
    def add(self,state,action,reward,next_state,done):
        """Add a new experience to memory."""
        e = self.experience(state,action,reward,next_state,done)
        self.memory.append(e)
        
    def sample(self, num_agents):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory,k = self.batch_size)
        states = [torch.from_numpy(np.vstack([e.states[index] for e in experiences if e is not None])).float().to(device) for index in range(num_agents)]
        actions = [torch.from_numpy(np.vstack([e.actions[index] for e in experiences if e is not None])).float().to(device) for index in range(num_agents)]
        rewards = [torch.from_numpy(np.vstack([e.rewards[index] for e in experiences if e is not None])).float().to(device) for index in range(num_agents)]
        next_states = [torch.from_numpy(np.vstack([e.next_states[index] for e in experiences if e is not None])).float().to(device) for index in range(num_agents)]
        dones = [torch.from_numpy(np.vstack([e.dones[index] for e in experiences if e is not None]).astype(np.uint8)).float().to(device) for index in range(num_agents)]
        return (states,actions,rewards,next_states,dones)
    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
        
class MADDPG():
    """Wrapper class managing different agents in the environment."""
    
    def __init__(self, num_agents, state_size, action_size):
        """Initialize a MADDPGAgent wrapper.
        Params
        ======
            num_agents (int): the number of agents in the environment
            state_size (int): dimension of each state
            action_size (int): dimension of each action
        """
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)
        self.agents = [DDPG_Agent(self.state_size,self.action_size,num_agents,x)  for x in range(num_agents)]
        
        
    def reset(self):
        """Resets OU Noise for each agent."""
        for agent in self.agents:
            agent.reset()
    
    def act(self,state,add_noise = True):
        """Picks an action for each agent given their individual observations 
        and the current policy."""
        action = np.zeros([self.num_agents, self.action_size])
        for index,agent in enumerate(self.agents):
            action[index,:] = agent.act(state[index],add_noise = add_noise)
        return action
    
    def step(self,states,actions,rewards,next_states,dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        self.memory.add(states,actions,rewards,next_states,dones)
        
        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample(self.num_agents)
            for index,agent in enumerate(self.agents):
                agent.learn(experiences, self.num_agents)
        
    def save(self):
        """Save Weights of every agent"""
        for index,agent in enumerate(self.agents):
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor_'+str(index)+ '.pth')
            torch.save(agent.critic_local.state_dict(),'checkpoint_critic_'+str(index)+ '.pth')
            torch.save(agent.actor_target.state_dict(), 'checkpoint_actor_target_'+str(index)+ '.pth')
            torch.save(agent.critic_target.state_dict(),'checkpoint_critic_target_'+str(index)+ '.pth')
        
        