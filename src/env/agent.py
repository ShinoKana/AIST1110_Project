from abc import ABC, abstractmethod
import pickle
import sys
import numpy as np
import random
from typing import Tuple

import gym
import math
from collections import defaultdict
from itertools import count

import matplotlib.pyplot as plt

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
    def __init__(self, **kwargs):
        super().__init__()
        self.layout = kwargs['layout']
        self.q_table_file = ''.join(['agent_', self.layout, '.pkl'])
        self.q_table = None
        self.render_mode = kwargs['mode']

    def act(self, **kwargs):
        if self.q_table is None:
            try:
                with open(self.q_table_file, 'rb') as fp:
                    self.q_table = pickle.load(fp)
            except FileNotFoundError:
                print("Q table not found")
                sys.exit(1)

        matrix = kwargs['matrix']
        state = Agent.get_state(kwargs['player_pos'])

        try:
            if np.argmax(self.q_table[state]) == 0:
                if state[0] == 0:
                    if np.argmax(self.q_table[(len(matrix[0])-1, state[1])]) == 1:
                        return random.choice([1,2,3])
                    elif np.argmax(self.q_table[(state[0] - 1, state[1])]) == 1:
                        return random.choice([1,2,3])
            if np.argmax(self.q_table[state]) == 1:
                if state[0] == len(matrix[0])-1:
                    if np.argmax(self.q_table[(0, state[1])]) == 0:
                        return random.choice([0,2,3])
                    elif np.argmax(self.q_table[(state[0] + 1, state[1])]) == 0:
                        return random.choice([0,2,3])
            if np.argmax(self.q_table[state]) == 2:
                if state[1] == 0:
                    if np.argmax(self.q_table[(state[0], len(matrix)-1)]) == 3:
                        return random.choice([0,1,3])
                    elif np.argmax(self.q_table[(state[0], state[1] - 1)]) == 3:
                        return random.choice([0,1,3])
            if np.argmax(self.q_table[state]) == 3:
                if state[1] == len(matrix)-1:
                    if np.argmax(self.q_table[(state[0], 0)]) == 2:
                        return random.choice([0,1,2])
                    elif np.argmax(self.q_table[(state[0], state[1] + 1)]) == 2:
                        return random.choice([0,1,2])
            return np.argmax(self.q_table[state])
        except KeyError:
            print("KeyError")
            return random.randint(0, 3)
            
    def neighbors(i,j, matrix):
        tmp = [(i-1,j),(i+1,j),(i,j-1),(i,j+1)]
        if i == 0:
            tmp.append((len(matrix[0])-1,j))
        if i == len(matrix[0])-1:
            tmp.append((0,j))
        if j == 0:
            tmp.append((i,len(matrix)-1))
        if j == len(matrix)-1:
            tmp.append((i,0))

        return tmp

    def get_state(player_position: Tuple[int, int]):
        return player_position[0], player_position[1]

    def train(self, episodes):
        n_episodes = episodes
        discount = 0.99
        epsilon = 1.0
        epsilon_min = 0.1
        epsilon_decay_rate = 2 * 1e3 # designed for 10000 episodes
        print("Training mode ", self.render_mode)
        env = gym.make('pacman-v0', layout=self.layout)
        env = SkipFrame(env, skip=5)
        q_table = defaultdict(lambda: np.zeros(env.action_space.n))
        state = Agent.get_state(env.game.maze.get_player_home())

        plt.style.use('ggplot')
        plt.figure(figsize=(8, 6))
        x = np.arange(0, n_episodes, 1)
        y = []

        for episode in range(n_episodes):
            info = env.reset(mode='info')
            total_rewards = 0

            # epsilon_now = 1 - 0.9 * episode / ( n_episodes -1 )
            epsilon_now = epsilon_min + (epsilon - epsilon_min) * math.exp((-1) * episode / epsilon_decay_rate)
            # plot the epsilon
            print(x[episode])
            print(epsilon_now)
            y.append(epsilon_now)

            for i in count():
                env.render(self.render_mode)
                matrix = env.get_state_matrix()

                if random.uniform(0, 1) > epsilon_now:
                    action = int(np.argmax(q_table[state]))
                else:
                    neighbors_now = Agent.neighbors(state[0], state[1], matrix)
                    for neighbor in neighbors_now:
                        if neighbor[0] < 0 or neighbor[0] >= len(matrix[0]) or neighbor[1] < 0 or neighbor[1] >= len(matrix):
                            neighbors_now.remove(neighbor)
                        # print(neighbor)
                        if matrix[neighbor[1]][neighbor[0]] == -1:
                            neighbors_now.remove(neighbor)
                    if len(neighbors_now) == 0:
                        action = random.randint(0, 3)
                    else:
                        choice = random.choice(neighbors_now)
                        if choice == (state[0] - 1, state[1]):
                            action = 0
                        if choice == (state[0] + 1, state[1]):
                            action = 1
                        if choice == (state[0], state[1] - 1):
                            action = 2
                        if choice == (state[0], state[1] + 1):
                            action = 3
                    
                obs, rewards, done, info = env.step(action)

                next_state = Agent.get_state(info['player position'])

                if next_state != state:
                    rewards = rewards + 2 if rewards > 0 else rewards
                    q_table[state][action] += 0.6 * (rewards + discount * np.max(q_table[next_state]) - q_table[state][action])
                            

                state = next_state
                total_rewards += rewards

                if done:
                    print(f'{episode} episode finished after {i} timesteps')
                    print(f'epsilon: {epsilon_now}')
                    print(f'Total rewards: {total_rewards}')
                    print(f'game score: {info["game score"]}')
                    print(f'win: {info["win"]}')
                    break

        env.close()
        with open(self.q_table_file, 'wb') as fp:
            pickle.dump(dict(q_table), fp)
        
        plt.plot(x, y)
        plt.xlabel('episode')
        plt.ylabel('epsilon')
        plt.show()

        