from abc import ABC, abstractmethod
import pickle
import sys
import numpy as np
import random
from typing import List, Tuple

import gym
import math
from collections import defaultdict
from itertools import count

from gym.envs.registration import register

register(
    id='pacman-v0',
    entry_point='pacman_env:PacmanEnv',
)

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

class Agent(ABC):
    name = 'agent'

    def __init__(self, **kwargs):
        super().__init__()
        self.layout = kwargs['layout']
        self.q_table_file = ''.join([self.name, '_', self.layout, '.pkl'])
        self.q_table = None

    def act(self, *args, **kwargs):
        """
        Given the state the agent return the action to take

        :param args: arguments
        :param kwargs: other argument passed by keyword
        :return: the action to take
        """
        if self.q_table is None:
            try:
                with open(self.q_table_file, 'rb') as fp:
                    self.q_table = pickle.load(fp)
            except FileNotFoundError:
                print("Q table not found")
                sys.exit(1)

        state = Agent.get_state(kwargs['player_pos'], kwargs['ghost_positions'])
        try:
            return np.argmax(self.q_table[state])
        except KeyError:
            return random.randint(0, 3)

    def get_state(player_position, ghosts_positions: List[Tuple[int, int]]):
        g_y, g_x = ghosts_positions[0][1], ghosts_positions[0][0]

        return player_position[0], player_position[1], g_x, g_y
        
    def train(self, episodes):
        """
        Train the agent

        :param kwargs:
        :return:
        """
        n_episodes = episodes
        discount = 0.99
        alpha = 0.6  # learning rate
        epsilon = 1.0
        epsilon_min = 0.1
        epsilon_decay_rate = 1e6
        env = gym.make('pacman-v0', layout=self.layout)
        env = SkipFrame(env, skip=10)
        grid = np.zeros((env.action_space.n))
        # q_table = dict(lambda: np.zeros(env.action_space.n))
        # 这里有改动
        q_table = defaultdict(lambda: np.zeros(env.action_space.n))
        state = Agent.get_state(env.game.maze.get_player_home(), env.get_state_matrix())

        # def epsilon_by_frame(frame_idx):
        #     result = epsilon_min + (epsilon - epsilon_min) * math.exp(-1. * frame_idx / 
        #              epsilon_decay_rate)

        epsilon_by_frame = lambda frame_idx: epsilon_min + (epsilon - epsilon_min) * math.exp(
            -1. * frame_idx / epsilon_decay_rate)

        for episode in range(n_episodes):
            env.reset()
            total_rewards = 0

            epsilon_now = epsilon_by_frame(episode)
            # epsilon_now = epsilon_min + (epsilon - epsilon_min) * math.exp(
            #               -1. * episode / epsilon_decay_rate)

            for i in count():
                env.render()
                if random.uniform(0, 1) > epsilon_now:
                    action = int(np.argmax(q_table[state]))
                else:
                    action = env.action_space.sample()

                obs, rewards, done, info = env.step(action)
                next_state = Agent.get_state(info['player position'], info['state matrix'])

                if next_state != state:
                    rewards = rewards + 2 if rewards > 0 else rewards
                    q_table[state][action] += alpha * (
                            rewards + discount * np.max(q_table[next_state]) - q_table[state][action])

                state = next_state
                total_rewards += rewards

                if done:
                    print(f'{episode} episode finished after {i} timesteps')
                    print(f'Total rewards: {total_rewards}')
                    print(f'win: {info["win"]}')
                    break

                # env.close()

                with open(self.q_table_file, 'wb') as fp:
                    pickle.dump(dict(q_table), fp) 

