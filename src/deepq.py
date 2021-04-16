#!/usr/bin/env python3

import numpy as np
import torch
from torch.nn import Linear, Softmax, Conv1d, Flatten
import torch.nn.functional as F

class DQN(torch.nn.Module):
    def __init__(self, n_board_inputs, n_tiles, n_actions, n_hidden):
        super(DQN, self).__init__()
        self.n_board_inputs = n_board_inputs
        self.n_tiles = n_tiles
        self.board_conv1 = Conv1d(1, 32, stride=2, padding=1, kernel_size=3)
        self.board_conv2 = Conv1d(32, 64, stride=2, padding=1, kernel_size=3)
        self.flat_board = Flatten()

        self.dense = Linear(n_board_inputs + n_tiles, n_hidden)
        self.hidden1 = Linear(n_hidden, n_hidden)
        self.hidden2 = Linear(n_hidden, n_hidden)
        self.classifier = Linear(n_hidden, n_actions)

    def forward(self, inputs):
        board = inputs[:, :self.n_board_inputs]
        tile = inputs[:, self.n_board_inputs:]
        board = board.unsqueeze(-2) # Add channel dimension
        board = self.board_conv1(board)
        board = self.board_conv2(board)
        board = self.flat_board(board)

        inp = torch.cat((board, tile), 1)
        # print(inp.shape)
        out = F.relu(self.dense(inputs))
        # print(out.shape)
        out = F.relu(self.hidden1(out))
        # print(out.shape)
        out = F.relu(self.hidden2(out))
        # print(out.shape)
        return self.classifier(out)
