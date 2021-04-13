import numpy as np
import random
import math
import h5py

import time
from datetime import datetime
from tensorboardX import SummaryWriter
from functools import reduce

from deepq import DQN
import torch

import random
from copy import deepcopy

# This file provides the skeleton structure for the classes TQAgent and TDQNAgent to be completed by you, the student.
# Locations starting with # TO BE COMPLETED BY STUDENT indicates missing code that should be written by you.
#
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
    # board_id = reduce(lambda a,b: 2*a+b, board_state.flatten().astype(bool))
    class_id = class_state << np.size(board_state)
    return board_id + class_id

# def decode_action(action):
#     """ Takes int, returns (column, rotation)."""
#     bits = format(action, '04b')
#     loc = 2 * int(bits[0]) + int(bits[1])
#     rot = 2 * int(bits[2]) + int(bits[3])
#     return (loc, rot)

class TQAgent:
    # Agent for learning to play tetris using Q-learning
    def __init__(self,alpha,epsilon,episode_count):
        # Initialize training parameters
        self.alpha=alpha
        self.epsilon=epsilon
        self.episode=0
        self.episode_count=episode_count

    def fn_init(self,gameboard, name=''):
        def compute_legal_masks(self):
            legal_masks = np.zeros((len(gameboard.tiles), self.max_num_actions), dtype=bool)
            for t in range(len(self.gameboard.tiles)):
                self.gameboard.cur_tile_type = t
                for act in range(self.max_num_actions):
                    loc, rot = self.action_store[act]
                    is_valid = not self.gameboard.fn_move(loc, rot)
                    legal_masks[t, act] = is_valid

            return legal_masks

        self.gameboard=gameboard
        self.experiment_name = name
        self.writer = SummaryWriter("./log/" + self.experiment_name + '_' + datetime.now().strftime("%Y%m%d-%H%M%S"))
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # In this function you could set up and initialize the states, actions and Q-table and storage for the rewards
        # This function should not return a value, store Q table etc as attributes of self

        # Useful variables: 
        # 'gameboard.N_row' number of rows in gameboard
        # 'gameboard.N_col' number of columns in gameboard
        # 'len(gameboard.tiles)' number of different tiles
        # 'self.episode_count' the total number of episodes in the training

        self.reward_tots = np.zeros(self.episode_count)
        self.max_num_actions = self.gameboard.N_col * 4
        self.num_states = 2**(self.gameboard.N_row * self.gameboard.N_col + int(np.log2(self.max_num_actions)))
        self.q_table = np.zeros((self.num_states, self.max_num_actions), dtype=np.float32)

        # Initialize curr state
        # self.fn_read_state()

        # Legal move matrix (entry for each piece, masks every possible action)
        self.action_store = create_action_store(self.gameboard.N_col, 4)
        self.legal_masks = compute_legal_masks(self)

    def fn_load_strategy(self,strategy_file):
        # TO BE COMPLETED BY STUDENT
        # Here you can load the Q-table (to Q-table of self) from the input parameter strategy_file (used to test how the agent plays)
        self.q_table = np.load(strategy_file)


    def fn_read_state(self):
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # In this function you could calculate the current state of the game board
        # You can for example represent the state as an integer entry in the Q-table
        # This function should not return a value, store the state as an attribute of self

        # Useful variables: 
        # 'self.gameboard.N_row' number of rows in gameboard
        # 'self.gameboard.N_col' number of columns in gameboard
        # 'self.gameboard.board[index_row,index_col]' table indicating if row 'index_row' and column 'index_col' is occupied (+1) or free (-1)
        # 'self.gameboard.cur_tile_type' identifier of the current tile that should be placed on the game board (integer between 0 and len(self.gameboard.tiles))
        board_state = self.gameboard.board
        class_state = self.gameboard.cur_tile_type
        self.curr_state = encode_state(board_state, class_state)


    def fn_select_action(self):
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # Choose and execute an action, based on the Q-table or random if epsilon greedy
        # This function should not return a value, store the action as an attribute of self and exectute the action by moving the tile to the desired position and orientation

        # Useful variables: 
        # 'self.epsilon' parameter epsilon in epsilon-greedy policy

        # Useful functions
        # 'self.gameboard.fn_move(tile_x,tile_orientation)' use this function to execute the selected action
        # The input argument 'tile_x' contains the column of the tile (0 < tile_x < self.gameboard.N_col)
        # The input argument 'tile_orientation' contains the number of 90 degree rotations of the tile (0 < tile_orientation < # of non-degenerate rotations)
        # The function returns 1 if the action is not valid and 0 otherwise
        # You can use this function to map out which actions are valid or not

        # epsilon-greedy
        legal_mask = self.legal_masks[self.gameboard.cur_tile_type]
        legal_actions = np.arange(self.max_num_actions)[legal_mask]
        if np.random.rand() < self.epsilon:
            self.curr_action = np.random.choice(legal_actions)
        else:
            options = self.q_table[self.curr_state][legal_mask]
            winner = np.random.choice(np.argwhere(options == np.amax(options)).flatten())
            self.curr_action = legal_actions[winner]
            # self.curr_action = legal_actions[np.argmax(self.q_table[self.curr_state][legal_mask])]

        # Decode action
        loc, rot = self.action_store[self.curr_action]
        self.gameboard.fn_move(loc, rot)

    def fn_reinforce(self, old_state, reward):
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # Update the Q table using state and action stored as attributes in self and using function arguments for the old state and the reward
        # This function should not return a value, the Q table is stored as an attribute of self

        # Useful variables:
        # 'self.alpha' learning rate

        actions = np.arange(self.max_num_actions)
        legal_mask = self.legal_masks[self.gameboard.cur_tile_type]
        legal_actions = actions[legal_mask]
        options = self.q_table[self.curr_state][legal_mask]
        optimal_action = legal_actions[np.argmax(options)]

        d_temporal = reward + self.q_table[self.curr_state, optimal_action] - self.q_table[old_state, self.curr_action]
        self.q_table[old_state, self.curr_action] = self.q_table[old_state, self.curr_action] + self.alpha * d_temporal


    def fn_turn(self):
        if self.gameboard.gameover:
            self.writer.add_scalar("q_learning_agent/reward", self.reward_tots[self.episode], self.episode)
            self.episode+=1
            if self.episode%100==0:
                print('episode '+str(self.episode)+'/'+str(self.episode_count)+' (reward: ',str(np.sum(self.reward_tots[range(self.episode-100,self.episode)])),')')
                self.writer.add_scalar("q_learning_agent/average", self.reward_tots[self.episode-100:].sum(), self.episode)
            if self.episode%1000==0:
                saveEpisodes=[1000,2000,5000,10000,20000,50000,100000,200000,500000,1000000];
                if self.episode in saveEpisodes:
                    # TO BE COMPLETED BY STUDENT
                    # Here you can save the rewards and the Q-table to data files for plotting of the rewards and the Q-table can be used to test how the agent plays
                    np.save(f'log/{self.experiment_name}_{self.episode}_q', self.q_table)
                    np.save(f'log/{self.experiment_name}_{self.episode}_rewards', self.reward_tots)

            if self.episode>=self.episode_count:
                raise SystemExit(0)
            else:
                self.gameboard.fn_restart()
        else:
            # Select and execute action (move the tile to the desired column and orientation)
            self.fn_select_action()
            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to copy the old state into the variable 'old_state' which is later passed to fn_reinforce()
            old_state = self.curr_state

            # Drop the tile on the game board
            reward=self.gameboard.fn_drop()
            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to add the current reward to the total reward for the current episode, so you can save it to disk later
            self.reward_tots[self.episode] += reward

            # Read the new state
            self.fn_read_state()
            # Update the Q-table using the old state and the reward (the new state and the taken action should be stored as attributes in self)
            self.fn_reinforce(old_state,reward)

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.storage = []
        self.current = 0

    def __len__(self):
        return len(self.storage)

    def push(self, element):
        if len(self.storage) < self.capacity:
            self.storage.append(None)
        self.storage[self.current] = element
        self.current = (self.current + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.storage, batch_size)



class TDQNAgent:
    # Agent for learning to play tetris using Q-learning
    def __init__(self,alpha,epsilon,epsilon_scale,replay_buffer_size,batch_size,sync_target_episode_count,episode_count):
        # Initialize training parameters
        self.alpha=alpha
        self.epsilon=epsilon
        self.epsilon_scale=epsilon_scale
        self.replay_buffer_size=replay_buffer_size
        self.batch_size=batch_size
        self.sync_target_episode_count=sync_target_episode_count
        self.episode=0
        self.episode_count=episode_count

    def fn_init(self,gameboard,name=''):
        self.gameboard=gameboard
        self.experiment_name = name
        self.writer = SummaryWriter("./log/" + self.experiment_name + '_' + datetime.now().strftime("%Y%m%d-%H%M%S"))
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # In this function you could set up and initialize the states, actions, the Q-networks (one for calculating actions and one target network), experience replay buffer and storage for the rewards
        # You can use any framework for constructing the networks, for example pytorch or tensorflow
        # This function should not return a value, store Q network etc as attributes of self

        # Useful variables: 
        # 'gameboard.N_row' number of rows in gameboard
        # 'gameboard.N_col' number of columns in gameboard
        # 'len(gameboard.tiles)' number of different tiles
        # 'self.alpha' the learning rate for stochastic gradient descent
        # 'self.episode_count' the total number of episodes in the training
        # 'self.replay_buffer_size' the number of quadruplets stored in the experience replay buffer

        self.reward_tots = np.zeros(self.episode_count)
        self.max_num_actions = self.gameboard.N_col * 4
        self.chosen_action = False

        self.action_network = DQN(np.size(self.gameboard.board), len(self.gameboard.tiles), self.max_num_actions, 64)
        self.optimizer = torch.optim.Adam(self.action_network.parameters(), lr=self.alpha)
        self.target_network = deepcopy(self.action_network)
        self.target_network.eval()

        self.exp_buffer = ReplayBuffer(self.replay_buffer_size)
        self.action_store = create_action_store(self.gameboard.N_col, 4)

    def fn_load_strategy(self,strategy_file):
        self.action_network.load_state_dict(torch.load(strategy_file))
        self.target_network = deepcopy(self.action_network)
        self.target_network.eval()

    # TO BE COMPLETED BY STUDENT
    # Here you can load the Q-network (to Q-network of self) from the strategy_file

    def fn_read_state(self):
    # TO BE COMPLETED BY STUDENT
    # This function should be written by you
    # Instructions:
    # In this function you could calculate the current state of the gane board
    # You can for example represent the state as a copy of the game board and the identifier of the current tile
    # This function should not return a value, store the state as an attribute of self

        # Useful variables: 
        # 'self.gameboard.N_row' number of rows in gameboard
        # 'self.gameboard.N_col' number of columns in gameboard
        # 'self.gameboard.board[index_row,index_col]' table indicating if row 'index_row' and column 'index_col' is occupied (+1) or free (-1)
        # 'self.gameboard.cur_tile_type' identifier of the current tile that should be placed on the game board (integer between 0 and len(self.gameboard.tiles))
        board_state = self.gameboard.board.copy().flatten()
        class_state = np.zeros(len(self.gameboard.tiles), dtype=bool)
        class_state[self.gameboard.cur_tile_type] = True
        self.curr_state = np.concatenate([board_state, class_state]).astype(np.float32)

    def fn_select_action(self):
    # TO BE COMPLETED BY STUDENT
    # This function should be written by you
    # Instructions:
    # Choose and execute an action, based on the output of the Q-network for the current state, or random if epsilon greedy
    # This function should not return a value, store the action as an attribute of self and exectute the action by moving the tile to the desired position and orientation

        # Useful variables: 
        # 'self.epsilon' parameter epsilon in epsilon-greedy policy
        # 'self.epsilon_scale' parameter for the scale of the episode number where epsilon_N changes from unity to epsilon

        # Useful functions
        # 'self.gameboard.fn_move(tile_x,tile_orientation)' use this function to execute the selected action
        # The input argument 'tile_x' contains the column of the tile (0 < tile_x < self.gameboard.N_col)
        # The input argument 'tile_orientation' contains the number of 90 degree rotations of the tile (0 < tile_orientation < # of non-degenerate rotations)
        # The function returns 1 if the action is not valid and 0 otherwise
        # You can use this function to map out which actions are valid or not

        if np.random.rand() < max(self.epsilon, 1-self.episode / self.epsilon_scale) and not self.chosen_action:
            self.curr_action = np.random.choice(np.arange(self.max_num_actions))
            self.chosen_action = True
        else:
            # Run network
            qualities = self.action_network(torch.from_numpy(self.curr_state))
            self.curr_action = torch.argmax(qualities)

        # Apply action
        loc, rot = self.action_store[self.curr_action]
        self.gameboard.fn_move(loc, rot)

    def fn_reinforce(self,batch):
    # TO BE COMPLETED BY STUDENT
    # This function should be written by you
    # Instructions:
    # Update the Q network using a batch of quadruplets (old state, last action, last reward, new state)
    # Calculate the loss function by first, for each old state, use the Q-network to calculate the values Q(s_old,a), i.e. the estimate of the future reward for all actions a
    # Then repeat for the target network to calculate the value \hat Q(s_new,a) of the new state (use \hat Q=0 if the new state is terminal)
    # This function should not return a value, the Q table is stored as an attribute of self

        # Useful variables: 
        # The input argument 'batch' contains a sample of quadruplets used to update the Q-network

        # old_states = np.zeros((self.batch_size, 20), dtype=np.float32)
        # targets = np.zeros(self.batch_size)

        old_state_batch, action_batch, reward_batch, new_state_batch, nonfinal_mask = tuple(map(torch.Tensor, zip(*batch)))
        action_batch = action_batch.unsqueeze(1).long()
        nonfinal_mask = nonfinal_mask > 0

        next_state_qualities = torch.zeros(self.batch_size)
        target_quals, _ = torch.max(self.target_network(new_state_batch[nonfinal_mask]), dim=1)
        next_state_qualities[nonfinal_mask] = target_quals
        targets = next_state_qualities + reward_batch
        outputs = self.action_network(old_state_batch).gather(1, action_batch).flatten()

        loss = (outputs - targets).pow(2).sum()
        # loss = torch.nn.functional.mse_loss(outputs, targets)
        self.optimizer.zero_grad()
        loss.backward()

        # for param in self.action_network.parameters():
        #         param.grad.data.clamp_(-1, 1)

        self.optimizer.step()


    def fn_turn(self):
        if self.gameboard.gameover:
            self.writer.add_scalar("deepq_agent/reward", self.reward_tots[self.episode], self.episode)
            self.episode+=1
            if self.episode%100==0:
                print('episode '+str(self.episode)+'/'+str(self.episode_count)+' (reward: ',str(np.sum(self.reward_tots[range(self.episode-100,self.episode)])),')')
                self.writer.add_scalar("deepq_agent/average", self.reward_tots[self.episode-100:].sum(), self.episode)
            if self.episode%1000==0:
                saveEpisodes=[1000,2000,5000,10000,20000,50000,100000,200000,500000,1000000];
                if self.episode in saveEpisodes:
                # TO BE COMPLETED BY STUDENT
                # Here you can save the rewards and the Q-network to data files
                    torch.save(self.action_network.state_dict(), f'log/{self.experiment_name}_{self.episode}_q.pt')
                    np.save(f'log/{self.experiment_name}_{self.episode}_rewards', self.reward_tots)


            if self.episode>=self.episode_count:
                raise SystemExit(0)
            else:
                if (len(self.exp_buffer) >= self.replay_buffer_size) and ((self.episode % self.sync_target_episode_count)==0):
                # TO BE COMPLETED BY STUDENT
                # Here you should write line(s) to copy the current network to the target network
                    self.target_network.load_state_dict(self.action_network.state_dict())

                self.gameboard.fn_restart()
        else:
            # Select and execute action (move the tile to the desired column and orientation)
            self.fn_select_action()
            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to copy the old state into the variable 'old_state' which is later stored in the ecperience replay buffer
            old_state = self.curr_state.copy()

            # Drop the tile on the game board
            reward = self.gameboard.fn_drop()
            self.chosen_action = False

            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to add the current reward to the total reward for the current episode, so you can save it to disk later
            # self.
            self.reward_tots[self.episode] += reward

            # Read the new state
            self.fn_read_state()

            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to store the state in the experience replay buffer
            next_state = self.curr_state.copy()
            nonfinal_state = not self.gameboard.gameover
            entry = (old_state.copy(), self.curr_action, reward, next_state, nonfinal_state)
            self.exp_buffer.push(entry)

            if len(self.exp_buffer) >= self.replay_buffer_size:
                # TO BE COMPLETED BY STUDENT
                # Here you should write line(s) to create a variable 'batch' containing 'self.batch_size' quadruplets
                batch = self.exp_buffer.sample(self.batch_size)
                self.fn_reinforce(batch)

class THumanAgent:
    def fn_init(self,gameboard,name='human'):
        self.episode=0
        self.reward_tots=[0]
        self.gameboard=gameboard

    def fn_read_state(self):
        pass

    def fn_turn(self,pygame):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit(0)
            if event.type==pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.reward_tots=[0]
                    self.gameboard.fn_restart()
                if not self.gameboard.gameover:
                    if event.key == pygame.K_UP:
                        self.gameboard.fn_move(self.gameboard.tile_x,(self.gameboard.tile_orientation+1)%len(self.gameboard.tiles[self.gameboard.cur_tile_type]))
                    if event.key == pygame.K_LEFT:
                        self.gameboard.fn_move(self.gameboard.tile_x-1,self.gameboard.tile_orientation)
                    if event.key == pygame.K_RIGHT:
                        self.gameboard.fn_move(self.gameboard.tile_x+1,self.gameboard.tile_orientation)
                    if (event.key == pygame.K_DOWN) or (event.key == pygame.K_SPACE):
                        self.reward_tots[self.episode]+=self.gameboard.fn_drop()
