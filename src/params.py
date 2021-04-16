#!/usr/bin/env python3
# Game parameters:
# 'N_row' and 'N_col' (integers) gives the size of the game board.
# 'tile_size' (2 or 4) denotes whether the small tile set (2) or the large tile set (4) should be used
# 'max_tile_count' (integer) denotes the maximal number of tiles to be placed in one game
# 'stochastic_prob' (float between 0 and 1) denotes the probability to take a random tile. When stochastic_prob': 0 tiles are taken according to a predefined sequence, when stochastic_prob': 1 all tiles are random. For values 0<stochastic_prob<1 there is a mixture between deterministic and random tiles

# Training parameters:
# 'alpha' is learning rate in Q-learning or for the stochastic gradient descent in deep Q-networks
# 'epsilon' is probability to choose random action in epsilon-greedy policy
# 'episode_count' is the number of epsiodes a training session lasts

# Additional training parameters for deep Q-networks:
# 'epsilon_scale' is the scale of the episode number where epsilon_N changes from unity to epsilon
# 'replay_buffer_size' is the size of the experience replay buffer
# 'batch_size' is the number of samples taken from the experience replay buffer each update
# 'sync_target_episode_count' is the number of epsiodes between synchronisations of the target network
#
param_task1a = {
    'name': 'task1a',
    'strategy_file': '',
    'human_player': False,
    'evaluate_agent': False,
    'N_row': 4,
    'N_col': 4,
    'tile_size': 2,
    'max_tile_count': 50,
    'stochastic_prob': 0,
    'alpha': 0.2,
    'epsilon': 0,
    'episode_count': 1000,
    'use_deepq':  False,
}

param_task1b = {
    'name':  'task1b',
    'strategy_file': '',
    'human_player': False,
    'evaluate_agent': False,
    'N_row': 4,
    'N_col': 4,
    'tile_size': 2,
    'max_tile_count': 50,
    'stochastic_prob': 0,
    'alpha': 0.2,
    'epsilon': 0.001,
    'episode_count': 10000,
    'use_deepq':  False,
}
param_task1c =  {
    'name':  'task1c',
    'strategy_file': '',
    'human_player': False,
    'evaluate_agent': False,
    'N_row': 4,
    'N_col': 4,
    'tile_size': 2,
    'max_tile_count': 50,
    'stochastic_prob': 1,
    'alpha': 0.2,
    'epsilon': 0.001,
    'episode_count': 200000,
    'use_deepq':  False,
}
param_task1d =  {
    'name':  'task1d',
    'strategy_file': '',
    'human_player': False,
    'evaluate_agent': False,
    'N_row': 8,
    'N_col': 8,
    'tile_size': 4,
    'max_tile_count': 50,
    'stochastic_prob': 1,
    'alpha': 0.2,
    'epsilon': 0.001,
    'episode_count': 200000,
    'use_deepq':  False,
}

param_task2a =  {
    'name':  'task2a',
    'strategy_file': '',
    'human_player': False,
    'evaluate_agent': False,
    'N_row': 4,
    'N_col': 4,
    'tile_size': 2,
    'max_tile_count': 50,
    'stochastic_prob': 1,
    'alpha': 0.001,
    'epsilon': 0.001,
    'episode_count': 10000,
    'epsilon_scale': 5000,
    'replay_buffer_size': 10000,
    'batch_size': 32,
    'sync_target_episode_count': 100,
    'use_deepq':  True,
}
param_task2b =  {
    'name':  'task2b',
    'strategy_file': '',
    'human_player': False,
    'evaluate_agent': False,
    'N_row': 8,
    'N_col': 8,
    'tile_size': 4,
    'max_tile_count': 50,
    'stochastic_prob': 1,
    'alpha': 0.001,
    'epsilon': 0.001,
    'episode_count': 10000,
    'epsilon_scale': 50000,
    'replay_buffer_size': 10000,
    'batch_size': 32,
    'sync_target_episode_count': 100,
    'use_deepq':  True,
}

param_dict = {
    '1a': param_task1a,
    '1b': param_task1b,
    '1c': param_task1c,
    '1d': param_task1d,
    '2a': param_task2a,
    '2b': param_task2b,
}
