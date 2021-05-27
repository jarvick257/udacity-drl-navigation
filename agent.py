# import time
import pdb
import numpy as np

import model
from memory import ReplayBuffer

import torch


class DoubleDeepQAgent:
    def __init__(
        self,
        n_inputs,
        n_actions,
        lr,
        mem_size,
        batch_size,
        chkpt_dir,
        gamma=0.99,
        epsilon=1.0,
        eps_dec=1e-5,
        eps_min=0.005,
        replace=1000,
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.replace_target_counter = replace
        self.learn_step_counter = 0
        self.memory = ReplayBuffer(mem_size, n_inputs, n_actions)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.q_online = model.LinearDeepQNetwork(
            n_inputs=n_inputs,
            n_actions=self.n_actions,
            name="q_online",
            chkpt_dir=chkpt_dir,
        ).to(self.device)
        self.q_target = model.LinearDeepQNetwork(
            n_inputs=n_inputs,
            n_actions=self.n_actions,
            name="q_target",
            chkpt_dir=chkpt_dir,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.q_online.parameters(), lr=lr)
        self.loss = torch.nn.MSELoss()

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = torch.tensor([observation], dtype=torch.float).to(self.device)
            actions = self.q_online(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.randint(self.n_actions)
        return action

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def decrement_epsilon(self):
        self.epsilon = max(self.epsilon * self.eps_dec, self.eps_min)

    def sample_memory(self):
        experience = self.memory.sample_buffer(self.batch_size)
        experience = [torch.tensor(sars).to(self.device) for sars in experience]
        return experience

    def save_models(self):
        self.q_target.save_checkpoint()
        self.q_online.save_checkpoint()

    def load_models(self):
        self.q_target.load_checkpoint(self.device)
        self.q_online.load_checkpoint(self.device)

    def learn(self):
        if self.memory.mem_counter < self.batch_size:
            return

        # t0 = time.time()
        self.optimizer.zero_grad()

        if self.learn_step_counter % self.replace_target_counter == 0:
            self.q_target.load_state_dict(self.q_online.state_dict())
        # t1 = time.time()

        states, actions, rewards, states_, dones = self.sample_memory()

        # t2 = time.time()
        indices = np.arange(self.batch_size)
        q_pred = self.q_online(states)[indices, actions]
        with torch.no_grad():
            q_next = self.q_target(states_).max(dim=1)[0]
            q_next[dones] = 0.0
            q_target = rewards + self.gamma * q_next
        # print("target: ", q_target)
        # print("pred:   ", q_pred)
        # t3 = time.time()

        loss = self.loss(q_pred, q_target).to(self.device)
        loss.backward()
        self.optimizer.step()
        self.learn_step_counter += 1
        self.decrement_epsilon()
        # t4 = time.time()
        # print( f"step 1: {t1-t0:0.4f}, 2: {t2-t1:0.4f}, 3: {t3-t2:0.4f}, 4: {t4-t3:0.4f}")
