#!/usr/bin/env python3

import numpy as np
import torch
from torch.nn import Linear, Softmax, Conv1d, Flatten
import torch.nn.functional as F

# class DQN(torch.nn.Module):
#     def __init__(self, n_board_inputs, n_tiles, n_actions, n_hidden):
#         super(DQN, self).__init__()
#         self.dense = Linear(n_board_inputs + n_tiles, n_hidden)
#         self.hidden1 = Linear(n_hidden, n_hidden)
#         self.hidden2 = Linear(n_hidden, n_hidden)
#         self.classifier = Linear(n_hidden, n_actions)

#     def forward(self, inputs):
#         out = F.relu(self.dense(inputs))
#         out = F.relu(self.hidden1(out))
#         out = F.relu(self.hidden2(out))
#         return self.classifier(out)

class DQN(torch.nn.Module):
    def __init__(self, n_board_inputs, n_tiles, n_actions, n_hidden):
        super(DQN, self).__init__()
        self.n_board_inputs = n_board_inputs
        self.n_tiles = n_tiles
        self.conv1 = Conv1d(1, 8, stride=2, padding=1, kernel_size=3)
        self.conv2 = Conv1d(8, 16, stride=2, padding=1, kernel_size=3)
        self.flat_board = Flatten()

        self.dense = Linear(288, n_hidden)
        self.hidden1 = Linear(n_hidden, n_hidden)
        self.hidden2 = Linear(n_hidden, n_hidden)
        self.classifier = Linear(n_hidden, n_actions)

    def forward(self, inputs):
        # board = inputs[:, :self.n_board_inputs]
        # tile = inputs[:, self.n_board_inputs:]
        pre = inputs.unsqueeze(-2) # Add channel dimension
        pre = self.conv1(pre)
        pre = self.conv2(pre)
        pre = self.flat_board(pre)

        # inp = torch.cat((board, tile), 1)
        out = F.relu(self.dense(pre))
        # out = F.relu(self.hidden1(out))
        # out = F.relu(self.hidden2(out))
        return self.classifier(out)
