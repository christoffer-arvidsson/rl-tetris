#!/usr/bin/env python3

import numpy as np
import torch
from torch.nn import Linear, Softmax, Conv1d, Flatten
import torch.nn.functional as F

# class DQN(torch.nn.Module):
    # def __init__(self, n_board_inputs, n_tiles, n_actions, n_hidden):
    #     super(DQN, self).__init__()
    #     self.dense = Linear(n_board_inputs + n_tiles, n_hidden)
    #     self.n_hidden_layers = 2
    #     self.linears = torch.nn.ModuleList([
    #         Linear(n_hidden, n_hidden) for i in range(self.n_hidden_layers)
    #     ])
    #     # self.hidden1 = Linear(n_hidden, n_hidden)
    #     # self.hidden2 = Linear(n_hidden, n_hidden)
    #     self.classifier = Linear(n_hidden, n_actions)

    # def forward(self, inputs):
    #     out = F.relu(self.dense(inputs))
    #     for l in self.linears:
    #         out = F.relu(l(out))

    #     return self.classifier(out)

class DQN(torch.nn.Module):
    def __init__(self, n_board_inputs, n_tiles, n_actions, n_hidden):
        super(DQN, self).__init__()
        self.n_board_inputs = n_board_inputs
        self.n_tiles = n_tiles
        self.conv1 = Conv1d(1, 4, stride=2, kernel_size=5)
        self.conv2 = Conv1d(4, 8, stride=2, kernel_size=5)
        self.flat_board = Flatten()

        self.dense = Linear(120, n_hidden)
        self.n_hidden_layers = 2
        self.linears = torch.nn.ModuleList([
            Linear(n_hidden, n_hidden) for i in range(self.n_hidden_layers)
        ])
        self.classifier = Linear(n_hidden, n_actions)

    def forward(self, inputs):
        # board = inputs[:, :self.n_board_inputs]
        # tile = inputs[:, self.n_board_inputs:]
        out = inputs.unsqueeze(-2) # Add channel dimension
        out = F.relu(self.conv1(out))
        out = F.relu(self.conv2(out))
        out = self.flat_board(out)
        out = self.dense(out)

        for l in self.linears:
            out = F.relu(l(out))
        return self.classifier(out)
