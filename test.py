import sys
import os
import torch
import numpy as np

from unityagents import UnityEnvironment

from model import LinearDeepQNetwork

try:
    model_dir = sys.argv[1]
except IndexError:
    print(f"Usage: python {sys.argv[0]} <model_dir>")
    sys.exit(1)


score = 0
env = UnityEnvironment(file_name="Banana_Linux/Banana.x86_64")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=False)[brain_name]
action_size = brain.vector_action_space_size
state_size = len(env_info.vector_observations[0])

model = LinearDeepQNetwork(state_size, action_size, "q_online", model_dir)
model.load_checkpoint()
model.eval()

num_moves = 0
obs = env_info.vector_observations[0]
while not env_info.local_done[0]:
    state = torch.tensor([obs], dtype=torch.float)
    actions = model(state)
    action = torch.argmax(actions).item()
    env_info = env.step(action)[brain_name]
    obs = env_info.vector_observations[0]
    score += env_info.rewards[0]
    num_moves += 1

env.close()
print(f"Final score after {num_moves} moves: {score}")
