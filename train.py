import os
import numpy as np
from unityagents import UnityEnvironment
from agent import DoubleDeepQAgent
from utils import plot_learning_curve

lr = 0.001
n_games = 500
eps_history, scores = [], []
avg_score = -1e8
best_score = avg_score
chkpt_dir = f"models/BananaBrain_{n_games}_{lr}"
figure_file = os.path.join(chkpt_dir, "progress.png")

env = UnityEnvironment(file_name="Banana_Linux/Banana.x86_64")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]
action_size = brain.vector_action_space_size
print("Number of actions:", action_size)
state_size = len(env_info.vector_observations[0])
print("States have length:", state_size)

agent = DoubleDeepQAgent(
    n_inputs=state_size,
    n_actions=action_size,
    lr=lr,
    mem_size=100000,
    batch_size=64,
    replace=1000,
    epsilon=1.0,
    eps_dec=0.9998,
    gamma=0.99,
    chkpt_dir=chkpt_dir,
)

for i in range(n_games):
    score = 0
    done = False
    env_info = env.reset(train_mode=True)[brain_name]
    obs = env_info.vector_observations[0]

    while not done:
        action = agent.choose_action(obs)
        env_info = env.step(action)[brain_name]
        obs_ = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        agent.store_transition(obs, action, reward, obs_, done)
        agent.learn()
        obs = obs_
        score += reward

    eps_history.append(agent.epsilon)
    scores.append(score)

    if i > 0 and i % 100 == 0:
        avg_score = np.mean(scores[-100:])
        print(
            f"Episode {i} score {score :0.1f} avg score {avg_score :0.1f} epsilon {agent.epsilon :0.2f}"
        )
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()
            x = list(range(1, len(scores) + 1))
            plot_learning_curve(x, scores, eps_history, figure_file)

env.close()
