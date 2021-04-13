#!/usr/bin/env python3

import numpy as np
import torch
from torch.nn import Linear, Softmax
import torch.nn as nn
import torch.nn.functional as F

class DQN(torch.nn.Module):
    def __init__(self, board_width, board_height, n_tiles, n_actions, n_hidden):
        super(DQN, self).__init__()
        self.preprocess = nn.Sequential(
            nn.Conv2d(1,2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(2,4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(4,8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.dense = Linear(15, n_hidden)
        self.hidden1 = Linear(n_hidden, n_hidden)
        self.hidden2 = Linear(n_hidden, n_hidden)
        self.classifier = Linear(n_hidden, n_actions)

    def forward(self, inputs):
        board, tile = inputs
        board_encoding = self.preprocess(board).squeeze()

        x = torch.cat((board_encoding, tile.squeeze()),-1)
        x = F.relu(self.dense(x))
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        return self.classifier(x)
