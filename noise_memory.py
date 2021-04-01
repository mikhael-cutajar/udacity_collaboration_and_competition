# import random
# import copy
# from collections import deque
# from collections import namedtuple
# import numpy as np
# import torch

# class OUNoise():
#     """Ornstein-Uhlenbeck process"""

#     def __init__(self,size,mu = 0,theta = 0.15, sigma = 0.2):
#         """Initialize parameters and noise process."""
#         self.mu = mu * np.ones(size)
#         self.theta = theta
#         self.sigma = sigma
#         self.seed = random.seed(RANDOM_SEED)
#         self.reset()
#         self.size = size
        
#     def reset(self):
#         """Reset the internal state (= noise) to mean (mu)."""
#         self.state = copy.copy(self.mu)
        
#     def sample(self):
#         """Update internal state and return it as a noise sample."""
#         x = self.state
#         dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
#         self.state = x + dx        
#         return self.state
    

# class ReplayBuffer():
#     def __init__(self,buffer_size,batch_size):
#         """Initialize a ReplayBuffer object.
#         Params
#         ======
#             buffer_size (int): maximum size of buffer
#             batch_size (int): size of each training batch
#         """
#         self.batch_size = batch_size
#         self.memory = deque(maxlen = buffer_size)
#         self.batch_size = batch_size
#         self.experience = namedtuple("Experience",field_names= ["states","actions","rewards","next_states","dones"])
        
#     def add(self,state,action,reward,next_state,done):
#         """Add a new experience to memory."""
#         e = self.experience(state,action,reward,next_state,done)
#         self.memory.append(e)
        
#     def sample(self):
#         """Randomly sample a batch of experiences from memory."""
#         experiences = random.sample(self.memory, k = self.batch_size)
#         states = [torch.from_numpy(np.vstack([e.states[index] for e in experiences if e is not None])).float().to(device) for index in range(num_agents)]
#         actions = [torch.from_numpy(np.vstack([e.actions[index] for e in experiences if e is not None])).float().to(device) for index in range(num_agents)]
#         rewards = [torch.from_numpy(np.vstack([e.rewards[index] for e in experiences if e is not None])).float().to(device) for index in range(num_agents)]
#         next_states = [torch.from_numpy(np.vstack([e.next_states[index] for e in experiences if e is not None])).float().to(device) for index in range(num_agents)]
#         dones = [torch.from_numpy(np.vstack([e.dones[index] for e in experiences if e is not None]).astype(np.uint8)).float().to(device) for index in range(num_agents)]
#         return (states,actions,rewards,next_states,dones)
    
#     def __len__(self):
#         """Return the current size of internal memory."""
#         return len(self.memory)