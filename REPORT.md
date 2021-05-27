# Project Report

## Implementation Details
### agent.py
This file implements a double deep-q agent. The agent holds two q-networks, q_online and q_target, which are used to approximate the value function.
The agent also takes care of epsilon decay for epsilon-greedy exploration so that allows for good exploration at the start of the training while focusses on proven policies later on. There is however a min value for epsilon so that some exploration can always occur. Encountered experiences are stored in a simple replay buffer which is randomly sampled for training.

### memory.py
Implementation of a simple replay buffer without prioritization. This is a circular buffer with a specifiable size.

### model.py
This file defines the neural net to be used for the q-networks. As the problem at hand is rather simple, so is the network with only one hidden layer.

### utils.py
Helper functions are defined here. So far there is only one for plotting the training progress.

### train.py
Hyperparamters such as learning rate, epislon decay rate, number of games for training, etc. are all stored within this file. By default, the agent will be trained for 500 episodes. Every 100 episode, the average score of the agent is evaluated and, if this score is a new max, the trained weights are stored in `models/BananaBrain_[num_episodes]_[learningrate]`.
Other hyperparameters:
* `mem_size`: max size of the replay buffer - 100000 - set to some more or less random, high value. As the experiences are expected to be somewhat similar throughout the game, this should not matter too much.
* `batch_size`: batch size for training - 64 - typical value
* `replace`: number of training steps between updating `q_target` with the weights of `q_online` - 1000 - as one episode consists of ~300 moves, this will update the the weights about once every 3 episodes.
* `epsilon`,`eps_dec`,`eps_min`: Epsilon start-, decay- and min values - 1.0/0.998/0.005 - these values were chosen so that epsilon would decay roughly throughout the first 20% of training, resulting in decent exploration while still giving time to learn q values.
* `gamma`: discount factor for future vs immediate rewards - 1.0 - for the given task, there is no reason to discount future rewards.
* `lr`: learning rate - 0.001 - results in stable progress while still converging quickly.
* `n_games`: number of episodes to train - 500 - looking at the progress graph, we can see that after around 300 games progress slowed down significantly so theoretically 300 would do, too. But since the network is so small, training another 200 games for good measure doesn't take very long so we might as well do that. Also 500 is a nice number.

### test.py
This script demonstrates how well a trained agent is performing. It takes the folder in which trained networks are stored as a parameter (eg.: `python test.py ./models/BananaBrain_500_0.001`) and runs one episode at normal speed. At the end, the final score is printed.

