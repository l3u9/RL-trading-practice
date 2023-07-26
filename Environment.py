import os
import pandas as pd
import numpy as np
import random
import copy
from collections import deque
from datetime import datetime
from utils import *
from PPO import *


class CustomEnv:
    def __init__(self, df, initial_balance=1000, lookback_window_size=50):
        self.df = df
        self.df_total_steps = len(self.df) - 1
        # print(self.df_total_steps)
        self.initial_balance = initial_balance
        self.lookback_window_size = lookback_window_size
        
        self.action_space = np.array([0, 1, 2])
        
        self.orders_history = deque(maxlen=self.lookback_window_size)
        
        self.market_history = deque(maxlen=self.lookback_window_size)
        
        self.state_size = self.lookback_window_size * 10
        self.before_action = 0


    def reset(self, env_steps_size = 0):
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.crypto_held = 0
        self.crypto_sold = 0
        self.crypto_bought = 0
        self.episode_orders = 0
        self.env_steps_size = env_steps_size


        if env_steps_size > 0:
            self.start_step = random.randint(self.lookback_window_size, self.df_total_steps - env_steps_size)
            self.end_step = self.start_step + env_steps_size
        else:
            self.start_step = self.lookback_window_size
            self.end_step = self.df_total_steps
        
        self.current_step = self.start_step

        for i in reversed(range(self.lookback_window_size)):
            current_step = self.current_step - i
            self.orders_history.append([self.balance, self.net_worth, self.crypto_bought, self.crypto_sold, self.crypto_held])
            self.market_history.append([self.df.loc[current_step, 'open'],
                                       self.df.loc[current_step, 'high'],
                                       self.df.loc[current_step, 'low'],
                                       self.df.loc[current_step, 'close'],
                                       self.df.loc[current_step, 'volume']
                                       ])

        # print(np.array(self.orders_history).shape)
        # print(self.df.loc[current_step, 'open_time'])
        state = np.concatenate((self.market_history, self.orders_history), axis=1)
        return state


    def step(self, action):
        self.crypto_bought = 0
        self.crypto_sold = 0
        self.current_step += 1
        
        current_price = random.uniform(
            self.df.loc[self.current_step, 'open'],
            self.df.loc[self.current_step, 'close']
        )


        Date = int(self.df.loc[self.current_step, 'open_time'])

        Date = str(datetime.utcfromtimestamp(Date / 1000)).replace(" ", "_")


        
        if action == 0:
            pass
        
        elif action == 1: # buy
            if self.before_action != 2:
                self.crypto_bought = self.balance / current_price
                self.balance -= self.crypto_bought * current_price
                self.crypto_held += self.crypto_bought
            else:
                self.crypto_sold = self.crypto_held
                self.balance += self.crypto_sold * current_price
                self.crypto_held -= self.crypto_sold
            self.before_action = 1


        
        elif action == 2: # sell
            if self.before_action != 1:
                self.crypto_sold = self.crypto_held
                self.balance += self.crypto_sold * current_price
                self.crypto_held -= self.crypto_sold
            else:
                self.crypto_bought = self.balance / current_price
                self.balance -= self.crypto_bought * current_price
                self.crypto_held += self.crypto_bought
            self.before_action = 2

        
        # print(self.orders_history[-1])
        # Write_to_file(Date, self.orders_history[-1])
        
        self.prev_net_worth = self.net_worth
        self.net_worth = self.balance + self.crypto_held * current_price
        
        self.orders_history.append([self.balance, self.net_worth, self.crypto_bought, self.crypto_sold, self.crypto_held])
        
        reward = self.net_worth - self.prev_net_worth


        if self.net_worth <= self.initial_balance/2:
            done = True
        else:
            done = False
        
        obs = self._next_observation()
        
        return obs, reward, done


    def _next_observation(self):
        self.market_history.append([self.df.loc[self.current_step, 'open'],
                                self.df.loc[self.current_step, 'high'],
                                self.df.loc[self.current_step, 'low'],
                                self.df.loc[self.current_step, 'close'],
                                self.df.loc[self.current_step, 'volume']
                                ])


        obs = np.concatenate((self.market_history, self.orders_history), axis=1)

        return obs

    def render(self):
        pass