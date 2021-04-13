#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
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
        x = F.relu(self.dense(inputs))
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        return self.classifier(x)

class StateAutoEncoder(torch.nn.Module):
    def __init__(self, board_width, board_height):
        super(StateAutoEncoder, self).__init__()
        self.board_width = board_width
        self.board_height = board_height
        self.code_dim = board_width * board_height // 4
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.Flatten(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2),
            nn.ReLU(),
            nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2),
            nn.ReLU(),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
        )
    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        x = x.reshape([-1, 16, self.board_width // 8, self.board_height // 8])
        return self.decoder(x)

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)

        return x

input = torch.randn(32, 1, 8, 8)
ae = StateAutoEncoder(8,8)
print(ae.encoder(input).shape)
