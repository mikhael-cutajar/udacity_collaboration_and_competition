
[//]: # (Image References)


# Project 3: Collaboration and Competition
### The environment

In this project, a model using MADDPG was trained to solve the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1. In this implementation we will be working with 20 arm agents.


## The code

The code was implemented by adjusting [Udacity's ddpg pendulum implementation](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum) to work for two collaborating agents.

The process occurs as following:
At each episode:
  we firstly reset the agents and their scores.
  while the episode has not terminated:
    The agents perform an action givin the state they are in with regards to the current policy.
    The next state for each agent is then discovered and rewards are assigned.
    experience is saved in replay memory
    When enough experiences have been acquired, the actor and critic weights start being adjusted to reflect the information from the experience tuples.


### Model Architecture

The DDPG model architecture implemented used we two hidden layers with 256 units on both hidden layers. As opposed to the Udacity pendulum implementation we used batch normalisation on the critic. 

### Hyperparameters

The hyperparameters are identical to the Udacity pendulum implementation as they worked well.

```
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
```

### Training

Below shows the score and average score the agent achieved every 10 episodes. The increasing score value indicates the agent is learning about the environment and adapting their actions to maximise their score. This learning is achieved using the actor critic method of DDPG which learns both a q value function and a policy. This means that similar to a deep q network it uses a replay buffer. DDPG also makes use of soft updating the weights throughout the training process instead of hard updates at certain intervals.

The process took 161 episodes running on the udacity workspace with GPU enabled to achieve an average score of 30.13.

At first the model was refusing to learn never surpassing a score of 10 after several episodes, however the changes to the model architecture and the increase to max_t was sufficient to help the model beat the game.

```
Episode 10	Score: 0.75	Average Score: 0.41
Episode 20	Score: 0.76	Average Score: 0.59
Episode 30	Score: 2.89	Average Score: 1.17
Episode 40	Score: 3.47	Average Score: 1.72
Episode 50	Score: 5.23	Average Score: 2.36
Episode 60	Score: 9.20	Average Score: 3.265
Episode 70	Score: 8.56	Average Score: 3.27
Episode 80	Score: 25.05	Average Score: 5.18
Episode 90	Score: 36.15	Average Score: 8.06
Episode 100	Score: 37.04	Average Score: 10.84
Episode 110	Score: 35.11	Average Score: 14.42
Episode 120	Score: 34.98	Average Score: 17.85
Episode 130	Score: 32.30	Average Score: 20.91
Episode 140	Score: 36.82	Average Score: 24.13
Episode 150	Score: 35.20	Average Score: 27.16
Episode 160	Score: 34.04	Average Score: 29.85
Episode 161	Score: 35.71	Average Score: 30.13

Environment solved!
```

![Screenshot 2021-03-24 at 12 53 49](https://user-images.githubusercontent.com/74315440/112306329-fd29dd00-8c9f-11eb-81c7-5ab886e66298.png)

### Future Work

Possible future work could including: 
- Trying out the D4PG model and comparing the results
- Experiment with how learning changes when using even more agents
