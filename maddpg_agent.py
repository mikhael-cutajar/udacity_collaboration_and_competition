import torch.nn.functional as F
import torch.optim as optim
import torch.cuda
from collections import namedtuple, deque
import random
import numpy as np
import torch

from ddpg_agent import DDPG_Agent

BUFFER_SIZE = int(1e6)          # Replay buffer size
BATCH_SIZE = 256                # Batch size for training models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer():
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, random_seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen = buffer_size) # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",field_names= ["states","actions","rewards","next_states","dones"])
        self.random_seed = random.seed(random_seed)
        
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        
    def sample(self, num_agents):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory,k = self.batch_size)
        
        states = [torch.from_numpy(np.vstack([e.states[index] for e in experiences if e is not None])).float().to(device) for index in range(num_agents)]
        actions = [torch.from_numpy(np.vstack([e.actions[index] for e in experiences if e is not None])).float().to(device) for index in range(num_agents)]
        rewards = [torch.from_numpy(np.vstack([e.rewards[index] for e in experiences if e is not None])).float().to(device) for index in range(num_agents)]
        next_states = [torch.from_numpy(np.vstack([e.next_states[index] for e in experiences if e is not None])).float().to(device) for index in range(num_agents)]
        dones = [torch.from_numpy(np.vstack([e.dones[index] for e in experiences if e is not None]).astype(np.uint8)).float().to(device) for index in range(num_agents)]
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
        
class MADDPG():
    """Class managing the agents in the environment"""
    
    def __init__(self, num_agents, state_size, action_size, random_seed):
        """Initialize a MADDPG object.
        Params
        ======
            num_agents (int): the number of agents
            state_size (int): number of states
            action_size (int): number of actions
        """
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.random_seed = random.seed(random_seed)
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, random_seed)
        self.agents = [DDPG_Agent(self.state_size,self.action_size,num_agents,x,random_seed)  for x in range(num_agents)]

        
    def reset(self):
        """Resets Noise for each agent."""
        for agent in self.agents:
            agent.reset()
    
    def act(self,state,add_noise = True):
        """Pick action for each agent with correspondence to the current policy.
           Calls the act function from each agent in the environment"""
        action = np.zeros([self.num_agents, self.action_size])
        for index,agent in enumerate(self.agents):
            action[index,:] = agent.act(state[index],add_noise = add_noise)
        return action
    
    def step(self,states,actions,rewards,next_states,dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        self.memory.add(states, actions, rewards, next_states, dones)
        
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

        
        