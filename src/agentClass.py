import numpy as np
import random
import math
import h5py

import time
from datetime import datetime
from tensorboardX import SummaryWriter
from functools import reduce

from util import create_action_store, encode_state
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
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

    def fn_init(self, gameboard, name=''):
        def compute_legal_masks(self):
            tile = gameboard.cur_tile_type
            legal_masks = np.zeros((len(gameboard.tiles), self.max_num_actions), dtype=bool)
            for t in range(len(self.gameboard.tiles)):
                self.gameboard.cur_tile_type = t
                for act in range(self.max_num_actions):
                    loc, rot = self.action_store[act]
                    is_valid = not self.gameboard.fn_move(loc, rot)
                    legal_masks[t, act] = is_valid


            self.gameboard.cur_tile_type = tile
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
            # winner = np.random.choice(np.argwhere(options == np.amax(options)).flatten())
            winner = np.argmax(options)
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



class TDQNAgent:
    # Agent for learning to play tetris using Q-learning
    def __init__(self, params):
        # Initialize training parameters
        self.alpha=params['alpha']
        self.epsilon=params['epsilon']
        self.epsilon_scale=params['epsilon_linear_scale']
        self.epsilon_exponential_factor=params['epsilon_exponential_factor']
        self.epsilon_decay_method=params['epsilon_decay_method']
        self.epsilon_reward_thresholds=params['epsilon_reward_thresholds']
        self.initial_epsilon=params['initial_epsilon']
        self.replay_buffer_size=params['replay_buffer_size']
        self.replay_prioritization=params['replay_prioritization']
        self.initial_beta=params['initial_beta']
        self.batch_size=params['batch_size']
        self.discount_factor=params['discount_factor']
        self.sync_target_step_count=params['sync_target_step_count']
        self.episode=0
        self.step=0
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

            return legal_masks

        self.gameboard=gameboard
        self.experiment_name = name
        self.writer = SummaryWriter("./log/" + self.experiment_name + '_' + datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.reward_tots = np.zeros(self.episode_count)

        self.max_num_actions = self.gameboard.N_col * 4
        self.current_epsilon = self.initial_epsilon

        self.beta = self.initial_beta # Importance sampling

        self.action_network = DQN(np.size(self.gameboard.board), len(self.gameboard.tiles), self.max_num_actions, 64)
        self.optimizer = torch.optim.Adam(self.action_network.parameters(), lr=self.alpha)
        self.loss = torch.nn.MSELoss()
        self.target_network = deepcopy(self.action_network)
        self.target_network.eval()

        self.exp_buffer = PrioritizedReplayBuffer(self.replay_buffer_size, self.replay_prioritization)

        # Action decoding
        self.action_store = create_action_store(self.gameboard.N_col, 4)
        self.legal_masks = compute_legal_masks(self)
        self.legal_count = np.zeros((len(gameboard.tiles), self.max_num_actions), dtype=int)

        # Trick to make evaluation look properly
        self.chosen_action = True
        self.sync_step_count = 0

    def fn_load_strategy(self,strategy_file):
        self.action_network.load_state_dict(torch.load(strategy_file))
        self.target_network = deepcopy(self.action_network)
        self.target_network.eval()

    def fn_read_state(self):
        """Encode state by concatenating board and tile."""
        board_state = self.gameboard.board.copy().flatten()
        class_state = np.zeros(len(self.gameboard.tiles), dtype=bool)
        class_state[self.gameboard.cur_tile_type] = True
        self.curr_state = np.concatenate([board_state, class_state]).astype(np.float32)

    def update_epsilon(self):
        """Update the epsilon according to defined method. """
        if self.epsilon_decay_method == "linear":
            self.current_epsilon = max(self.epsilon, 1-self.episode / self.epsilon_scale)
        elif self.epsilon_decay_method == "exponential":
            self.current_epsilon = max(self.epsilon, self.epsilon_exponential_factor * self.current_epsilon)

    def fn_select_action(self):
        """Select action either by epsilon-greedy or from network. Only legal actions are selected."""
        legal_mask = self.legal_masks[self.gameboard.cur_tile_type]
        legal_actions = np.arange(self.max_num_actions)[legal_mask]

        # Epsilon-greedy
        if np.random.rand() < self.current_epsilon and not self.chosen_action and self.epsilon != 0:
            self.curr_action = np.random.choice(legal_actions)
            self.chosen_action = True
        else: # network
            curr_state = torch.from_numpy(self.curr_state).unsqueeze(0) # Add batch dim

            self.action_network.eval()
            with torch.no_grad():
                options = self.action_network(curr_state)[0, legal_mask]
            self.action_network.train()

            winner = torch.argmax(options)
            self.curr_action = legal_actions[winner]

        # Decode and apply action
        loc, rot = self.action_store[self.curr_action]
        invalid = self.gameboard.fn_move(loc, rot)

    def fn_reinforce(self,batch):
        """Reinforce batch of transitions."""
        # priority and importance sampling
        weights, idxs = batch[1:]
        weights = torch.tensor(weights)

        samples = batch[0]
        old_state_batch, action_batch, reward_batch, new_state_batch, nonfinal_mask, legal_masks = tuple(map(torch.Tensor, zip(*samples)))
        action_batch = action_batch.unsqueeze(1).long()
        nonfinal_mask = nonfinal_mask == True
        illegal_masks = legal_masks == False

        state_action_values = self.action_network(old_state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size)
        with torch.no_grad():
            next_state_values[nonfinal_mask], _ = torch.max(
                self.target_network(new_state_batch[nonfinal_mask]) - 1e5 * illegal_masks[nonfinal_mask],
                dim=1
            )

        expected_state_action_values = next_state_values + reward_batch
        td_error = (expected_state_action_values - state_action_values[:,0])
        loss = td_error.pow(2) * weights
        priorities = (torch.abs(td_error) + 1e-5).detach().numpy()
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.exp_buffer.update_priorities(idxs, priorities)
        self.optimizer.step()

        return loss

    def fn_turn(self):
        if self.gameboard.gameover:
            # Update params
            self.update_epsilon()
            self.beta = min(self.initial_beta + (self.episode+1) / self.episode_count, 1.0)

            # Logging
            self.writer.add_scalar("deepq_agent/reward", self.reward_tots[self.episode], self.episode)
            self.writer.add_scalar("deepq_agent/epsilon", self.current_epsilon, self.episode)
            self.writer.add_scalar("deepq_agent/priority_beta", self.beta, self.episode)

            self.episode+=1
            if self.episode >= 100:
                self.writer.add_scalar("deepq_agent/average100", self.reward_tots[self.episode-100:].sum() / 100, self.episode)
            if self.episode%100==0:
                print('episode '+str(self.episode)+'/'+str(self.episode_count)+' (reward: ',str(np.sum(self.reward_tots[range(self.episode-100,self.episode)])),')')
            if self.episode%5000==0:
                torch.save(self.action_network.state_dict(), f'log/{self.experiment_name}_{self.episode}_q.pt')
                np.save(f'log/{self.experiment_name}_{self.episode}_rewards', self.reward_tots)

            if self.episode>=self.episode_count:
                raise SystemExit(0)
            else:

                self.gameboard.fn_restart()
        else:
            self.fn_select_action()
            old_state = self.curr_state

            reward = self.gameboard.fn_drop()
            self.chosen_action = False
            self.reward_tots[self.episode] += reward

            self.fn_read_state()

            next_state = self.curr_state
            nonfinal_state = not self.gameboard.gameover
            entry = (old_state.copy(), self.curr_action, reward, next_state.copy(), nonfinal_state, self.legal_masks[self.gameboard.cur_tile_type])
            self.exp_buffer.push(entry)

            # Sample from replay buffer
            self.sync_step_count += 1
            if len(self.exp_buffer) >= self.replay_buffer_size:
                batch = self.exp_buffer.sample(self.batch_size, self.beta)
                loss = self.fn_reinforce(batch)
                self.writer.add_scalar("deepq_agent/loss", loss.item(), self.episode)

                if self.episode % self.sync_target_step_count == 0:
                    self.target_network.load_state_dict(self.action_network.state_dict())
                    self.sync_step_count = 0

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
