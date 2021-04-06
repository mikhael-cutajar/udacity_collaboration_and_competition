[//]: # (Image References)


# Project 3: Collaboration and Competition
### The environment

In this project, a model using MADDPG was trained to solve the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

## The code

The code was implemented by adjusting [Udacity's ddpg pendulum implementation](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum) to work for two collaborating agents.

The process occurs as following:
- At each episode:
  - we firstly reset the agents and their scores.
  - while the episode has not terminated:
    - The agents perform an action givin the state they are in with regards to the current policy.
    - The next state for each agent is then discovered and rewards are assigned.
    - experience is saved in replay memory
    - When enough experiences have been acquired, the actor and critic weights start being adjusted to reflect the information from the experience tuples.


### Model Architecture

The model architecture implemented used we two hidden layers with 512 units on the first hidden layer and 256 units on the second hidden layer. This was the same for both the actor and the critic. As opposed to the Udacity pendulum implementation we used batch normalisation on the critic. 

### Hyperparameters

```
BUFFER_SIZE = int(1e6)  # Replay buffer size
BATCH_SIZE = 256        # Batch size for training models
LR_ACTOR = 1e-4         # Learning rate of the actor 
LR_CRITIC = 1e-3        # Learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
TAU = 8e-2              # For soft update of target parameters
GAMMA = 0.99            # Discount Factor
```

### Training

Below shows the score and average score the agent achieved every 50 episodes. The increasing score value indicates the agent is learning about the environment and adapting their actions to maximise their score. The process took 1282 episodes running on the udacity workspace with GPU enabled to achieve an average score of 0.5162.

```
Episode 50	Average Score: 0.0080
Episode 100	Average Score: 0.0308
Episode 150	Average Score: 0.0552
Episode 200	Average Score: 0.0578
Episode 250	Average Score: 0.0694
Episode 300	Average Score: 0.1253
Episode 350	Average Score: 0.1905
Episode 400	Average Score: 0.2318
Episode 450	Average Score: 0.2229
Episode 500	Average Score: 0.1570
Episode 550	Average Score: 0.1124
Episode 600	Average Score: 0.0965
Episode 650	Average Score: 0.0977
Episode 700	Average Score: 0.1066
Episode 750	Average Score: 0.1255
Episode 800	Average Score: 0.1765
Episode 850	Average Score: 0.2192
Episode 900	Average Score: 0.2250
Episode 950	Average Score: 0.2100
Episode 1000	Average Score: 0.1977
Episode 1050	Average Score: 0.1838
Episode 1100	Average Score: 0.1777
Episode 1150	Average Score: 0.1725
Episode 1200	Average Score: 0.1990
Episode 1250	Average Score: 0.2668
Episode 1282	Average Score: 0.5162
```

![Screenshot 2021-04-06 at 12 03 57](https://user-images.githubusercontent.com/74315440/113694524-2d29a500-96d0-11eb-896e-60deda29d508.png)

### Future Work

Possible future work could include: 
- Experiment with how training changes with priority experience replay
- Adjust the model with techniques such as dropout and see how that affects learning
