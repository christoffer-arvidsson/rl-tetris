#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(torch.nn.Module):
    def __init__(self, n_board_inputs, n_tiles, n_actions, n_hidden):
        super(DQN, self).__init__()
        self.dense = nn.Linear(n_board_inputs + n_tiles, n_hidden)

        self.n_hidden_layers = 2
        self.linears = nn.ModuleList([
            nn.Linear(n_hidden, n_hidden) for i in range(self.n_hidden_layers)
        ])
        self.classifier = nn.Linear(n_hidden, n_actions)

    def forward(self, inputs):
        out = F.relu(self.dense(inputs))
        for l in self.linears:
            out = F.relu(l(out))

        return self.classifier(out)


# class DQN(torch.nn.Module):
#     def __init__(self, n_board_inputs, n_tiles, n_actions, n_hidden):
#         super(DQN, self).__init__()
#         self.n_board_inputs = n_board_inputs
#         self.n_tiles = n_tiles
#         self.width = int(np.sqrt(n_board_inputs))

#         self.conv1 = nn.Conv2d(1, 8, stride=2, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(8, 16, stride=2, kernel_size=3, padding=1)

#         self.flat_board = nn.Flatten()
#         self.dense = nn.Linear(20, n_hidden)
#         self.n_hidden_layers = 4
#         self.linears = nn.ModuleList([
#             nn.Linear(n_hidden, n_hidden) for i in range(self.n_hidden_layers)
#         ])
#         self.classifier = nn.Linear(n_hidden, n_actions)

#     def forward(self, inputs):
#         board = inputs[:, :self.n_board_inputs].reshape((-1, self.width, self.width))
#         board = board.unsqueeze(-3) # Add channel dimension
#         tile = inputs[:, self.n_board_inputs:]

#         out = F.relu(self.conv1(board))
#         out = F.relu(self.conv2(out))
#         out = self.flat_board(out)
#         # print(out.shape, tile.shape)
#         out = torch.cat((out, tile), axis=-1)
#         out = F.relu(self.dense(out))

#         for l in self.linears:
#             out = F.relu(l(out))
#         return self.classifier(out)
