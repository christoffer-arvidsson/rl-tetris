import numpy as np
import random
import math
import h5py

import time
from datetime import datetime
from tensorboardX import SummaryWriter
from functools import reduce

from util import create_action_store, encode_state
from deepq import DQN
import torch
import torch.nn.functional as F

import random
from copy import deepcopy

class TQAgent:
    # Agent for learning to play tetris using Q-learning
    def __init__(self, params):
        # Initialize training parameters
        self.alpha=params['alpha']
        self.epsilon=params['epsilon']
        self.episode=0
        self.episode_count=params['episode_count']

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

        self.reward_tots = np.zeros(self.episode_count)
        self.max_num_actions = self.gameboard.N_col * 4
        self.num_states = 2**(self.gameboard.N_row * self.gameboard.N_col + int(np.log2(self.max_num_actions)))
        self.q_table = np.zeros((self.num_states, self.max_num_actions), dtype=np.float32)

        self.action_store = create_action_store(self.gameboard.N_col, 4)
        self.legal_masks = compute_legal_masks(self)

    def fn_load_strategy(self,strategy_file):
        self.q_table = np.load(strategy_file)


    def fn_read_state(self):
        board_state = self.gameboard.board
        class_state = self.gameboard.cur_tile_type
        self.curr_state = encode_state(board_state, class_state)


    def fn_select_action(self):
        # epsilon-greedy
        legal_mask = self.legal_masks[self.gameboard.cur_tile_type]
        legal_actions = np.arange(self.max_num_actions)[legal_mask]
        if np.random.rand() < self.epsilon:
            self.curr_action = np.random.choice(legal_actions)
        else:
            options = self.q_table[self.curr_state][legal_mask]
            winner = np.random.choice(np.argwhere(options == np.amax(options)).flatten())
            self.curr_action = legal_actions[winner]

        # Decode action
        loc, rot = self.action_store[self.curr_action]
        self.gameboard.fn_move(loc, rot)

    def fn_reinforce(self, old_state, reward):
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
                    np.save(f'log/{self.experiment_name}_{self.episode}_q', self.q_table)
                    np.save(f'log/{self.experiment_name}_{self.episode}_rewards', self.reward_tots)

            if self.episode>=self.episode_count:
                raise SystemExit(0)
            else:
                self.gameboard.fn_restart()
        else:
            # Select and execute action (move the tile to the desired column and orientation)
            self.fn_select_action()
            old_state = self.curr_state

            # Drop the tile on the game board
            reward=self.gameboard.fn_drop()
            self.reward_tots[self.episode] += reward

            # Read the new state
            self.fn_read_state()
            # Update the Q-table using the old state and the reward (the new state and the taken action should be stored as attributes in self)
            self.fn_reinforce(old_state,reward)


class ReplayBuffer(object):
    """ Cyclic replay buffer. """
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
    def __init__(self, params):
        # Initialize training parameters
        self.alpha=params['alpha']
        self.epsilon=params['epsilon']
        self.epsilon_scale=params['epsilon_scale']
        self.replay_buffer_size=params['replay_buffer_size']
        self.batch_size=params['batch_size']
        self.sync_target_episode_count=params['sync_target_episode_count']
        self.episode=0
        self.episode_count=params['episode_count']


    def fn_init(self,gameboard,name=''):
        def compute_legal_masks(self):
            legal_masks = np.zeros((len(gameboard.tiles), self.max_num_actions), dtype=bool)
            for t in range(len(self.gameboard.tiles)):
                self.gameboard.cur_tile_type = t
                for act in range(self.max_num_actions):
                    loc, rot = self.action_store[act]
                    is_valid = not self.gameboard.fn_move(loc, rot)
                    legal_masks[t, act] = is_valid
                    legal_masks[t, act] = True


            return legal_masks

        self.gameboard=gameboard
        self.experiment_name = name

        self.writer = SummaryWriter("./log/" + self.experiment_name + '_' + datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.reward_tots = np.zeros(self.episode_count)

        self.max_num_actions = self.gameboard.N_col * 4

        self.action_network = DQN(np.size(self.gameboard.board), len(self.gameboard.tiles), self.max_num_actions, 64)
        self.optimizer = torch.optim.Adam(self.action_network.parameters(), lr=self.alpha)
        self.loss = torch.nn.MSELoss()
        self.target_network = deepcopy(self.action_network)
        self.target_network.eval()

        self.exp_buffer = ReplayBuffer(self.replay_buffer_size)
        self.action_store = create_action_store(self.gameboard.N_col, 4)
        self.legal_masks = compute_legal_masks(self)

    def fn_load_strategy(self,strategy_file):
        self.action_network.load_state_dict(torch.load(strategy_file))
        self.target_network = deepcopy(self.action_network)
        self.target_network.eval()

    def fn_read_state(self):
        board_state = self.gameboard.board.copy().flatten()
        class_state = np.zeros(len(self.gameboard.tiles), dtype=bool)
        class_state[self.gameboard.cur_tile_type] = True
        self.curr_state = np.concatenate([board_state, class_state]).astype(np.float32)

    def fn_select_action(self):
        legal_mask = self.legal_masks[self.gameboard.cur_tile_type]
        legal_actions = np.arange(self.max_num_actions)[legal_mask]

        if np.random.rand() < max(self.epsilon, 1-self.episode / self.epsilon_scale):
            # self.curr_action = np.random.choice(np.arange(self.max_num_actions))
            self.curr_action = np.random.choice(legal_actions)
        else:
            # Run network
            curr_state = torch.from_numpy(self.curr_state).unsqueeze(0) # Add batch dim
            self.action_network.eval()
            with torch.no_grad():
                options = self.action_network(curr_state)[0, legal_mask]
            # print(options)

            self.action_network.train()

            winner = torch.argmax(options)
            self.curr_action = legal_actions[winner]
            # self.curr_action = winner

        # Apply action
        loc, rot = self.action_store[self.curr_action]
        invalid = self.gameboard.fn_move(loc, rot)
        # if invalid:
        #     print("INVALID")


    def fn_reinforce(self,batch):
        old_state_batch, action_batch, reward_batch, new_state_batch, nonfinal_mask = tuple(map(torch.Tensor, zip(*batch)))
        action_batch = action_batch.unsqueeze(1).long()
        nonfinal_mask = nonfinal_mask > 0

        state_action_values = self.action_network(old_state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size)
        next_state_values[nonfinal_mask] = torch.max(self.target_network(new_state_batch[nonfinal_mask]), dim=1)[0]
        expected_state_action_values = next_state_values + reward_batch

        loss = self.loss(expected_state_action_values.unsqueeze(1), state_action_values)
        # loss = (state_action_values - expected_state_action_values.unsqueeze(1)).pow(2).mean()
        # loss = (expected_state_action_values.unsqueeze(1) - state_action_values).pow(2).mean()
        # loss = F.smooth_l1_loss(expected_state_action_values.unsqueeze(1), state_action_values)


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.writer.add_scalar("deepq_agent/loss", loss.item(), self.episode)


    def fn_turn(self):
        if self.gameboard.gameover:
            self.writer.add_scalar("deepq_agent/reward", self.reward_tots[self.episode], self.episode)

            epsilon = max(self.epsilon, 1-self.episode / self.epsilon_scale)
            self.writer.add_scalar("deepq_agent/epsilon", epsilon, self.episode)

            self.episode+=1
            if self.episode%100==0:
                print('episode '+str(self.episode)+'/'+str(self.episode_count)+' (reward: ',str(np.sum(self.reward_tots[range(self.episode-100,self.episode)])),')')
                self.writer.add_scalar("deepq_agent/sum100", self.reward_tots[self.episode-100:].sum(), self.episode)
                self.writer.add_scalar("deepq_agent/average100", self.reward_tots[self.episode-100:].mean(), self.episode)
            if self.episode%1000==0:
                saveEpisodes=[1000,2000,5000,10000,20000,50000,100000,200000,500000,1000000];
                if self.episode in saveEpisodes:
                    torch.save(self.action_network.state_dict(), f'log/{self.experiment_name}_{self.episode}_q.pt')
                    np.save(f'log/{self.experiment_name}_{self.episode}_rewards', self.reward_tots)


            if self.episode>=self.episode_count:
                raise SystemExit(0)
            else:
                if (len(self.exp_buffer) >= self.replay_buffer_size) and ((self.episode % self.sync_target_episode_count)==0):
                    self.target_network.load_state_dict(self.action_network.state_dict())

                self.gameboard.fn_restart()
        else:
            self.fn_select_action()
            old_state = self.curr_state.copy()
            reward = self.gameboard.fn_drop()

            self.reward_tots[self.episode] += reward

            self.fn_read_state()
            next_state = self.curr_state.copy()
            nonfinal_state = not self.gameboard.gameover
            entry = (old_state.copy(), self.curr_action, reward, next_state.copy(), nonfinal_state)
            self.exp_buffer.push(entry)

            if len(self.exp_buffer) >= self.replay_buffer_size:
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
