import os
import torch
import torch.nn.functional as F
import numpy as np


class LinearDeepQNetwork(torch.nn.Module):
    def __init__(self, n_inputs, n_actions, name, chkpt_dir):
        super(LinearDeepQNetwork, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, name)
        self.dense1 = torch.nn.Linear(n_inputs, 25)
        self.dense2 = torch.nn.Linear(25, n_actions)
        if not os.path.isdir(chkpt_dir):
            os.mkdir(chkpt_dir)

    def forward(self, inputs):
        x = F.relu(self.dense1(inputs))
        x = self.dense2(x)
        return x

    def save_checkpoint(self):
        print(f"... saving checkpoint {self.checkpoint_file} ...")
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self, device="cpu"):
        print(f"... loading checkpoint {self.checkpoint_file} ...")
        self.load_state_dict(torch.load(self.checkpoint_file, map_location=device))
