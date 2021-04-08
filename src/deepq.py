#!/usr/bin/env python3

import numpy as np
import torch
from torch.nn import Linear, Softmax
import torch.nn.functional as F

class DQN(torch.nn.Module):
    def __init__(self, n_board_inputs, n_tiles, n_actions, n_hidden):
        super(DQN, self).__init__()
        self.dense = Linear(n_board_inputs + n_tiles, n_hidden)
        self.hidden1 = Linear(n_hidden, n_hidden)
        self.hidden2 = Linear(n_hidden, n_hidden)
        self.classifier = Linear(n_hidden, n_actions)

    def forward(self, inputs):
        out = F.relu(self.dense(inputs))
        out = F.relu(self.hidden1(out))
        out = F.relu(self.hidden2(out))
        return F.softmax(self.classifier(out))
