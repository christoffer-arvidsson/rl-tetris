#!/usr/bin/env python3
import numpy as np

def create_action_store(num_columns, num_rotations):
    store = np.zeros((num_columns * num_rotations, 2), dtype=int)
    for loc in range(num_columns):
        for rot in range(num_rotations):
            store[rot + loc * num_rotations, 0] = loc
            store[rot + loc * num_rotations, 1] = rot

    return store

def encode_state(board_state, class_state):
    binary_rep = np.power(2, np.arange(0,np.size(board_state)), dtype=int)
    board_id = (binary_rep * board_state.flatten()).sum(dtype=int)
    class_id = class_state << np.size(board_state)
    return board_id + class_id

