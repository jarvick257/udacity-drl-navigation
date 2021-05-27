![Trained Agent](https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif)

# Project 1: Navigation
This is my solution to the Project 1 - Navigation of Udacity's Deep Reinforcement Learning nanodegree.

### Project Description
The goal is to navigate a robot on a square plane and collect as many yellow bananas as possible while avoiding blue bananas. For each frame, the environment provides a vector of 37 values containing the agent's velocity and ray-based perception of objects around agent's forward direction. The agent then has to choose the best out of four possible actions: move forward or backward or turn left or right.
The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

### Setup
The setup for this repo is identical to https://github.com/udacity/deep-reinforcement-learning#dependencies so make sure to follow the installation instructions there.

### Implementation
See the [report](REPORT.md) for a detailed description of the implementation and design choices.

### Artifacts
Training an agent will create a folder with name `models/BananaBrain_[num_games]_[learning_rate]` in which a visualization of the training progress will be stored together with the best performing network weights `q_target` and `q_online`.

![progress](https://raw.githubusercontent.com/jarvick257/udacity-drl-navigation/develop/models/BananaBrain_500_0.001/progress.png)

