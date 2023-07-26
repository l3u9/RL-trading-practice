import os
import pandas as pd
import numpy as np
import random
import copy
from collections import deque
from datetime import datetime
from utils import *
from model import *
from torch.optim import Adam, RMSprop
from model import Actor_Model, Critic_Model


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
#########################################################
        self.lr = 0.0001
        self.epochs = 1
        self.normalize_value = 100000
        self.optimizer = RMSprop
        self.Actor = Actor_Model(input_shape=self.state_size, action_space = self.action_space.shape[0], lr=self.lr, optimizer = self.optimizer)
        self.Critic = Critic_Model(input_shape=self.state_size, action_space = self.action_space.shape[0], lr=self.lr, optimizer = self.optimizer)
    # def create_writer(self):
    #     self.replay_count = 0
    #     self.writer = SummaryWriter(comment="Crypto_trader")

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




    def get_gaes(self, rewards, dones, values, next_values, gamma = 0.99, lamda = 0.95, normalize=True):
        deltas = [r + gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]

        target = gaes + values
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return np.vstack(gaes), np.vstack(target)

    def replay(self, states, actions, rewards, predictions, dones, next_states):
        # reshape memory to appropriate shape for training
        states = np.vstack(states)
        next_states = np.vstack(next_states)
        actions = np.vstack(actions)
        predictions = np.vstack(predictions)

        # Compute discounted rewards
        #discounted_r = np.vstack(self.discount_rewards(rewards))

        # Get Critic network predictions 
        values = self.Critic.forward(states)
        next_values = self.Critic.forward(next_states)
        # Compute advantages
        #advantages = discounted_r - values
        # print("values: ", type(values))
        # print("next_value: ", type(next_values))
        # advantages, target = self.get_gaes(rewards, dones, np.squeeze(values), np.squeeze(next_values))
        advantages, target = self.get_gaes(rewards, dones, values.detach().numpy(), next_values.detach().numpy())
        '''
        pylab.plot(target,'-')
        pylab.plot(advantages,'.')
        ax=pylab.gca()
        ax.grid(True)
        pylab.show()
        '''
        # stack everything to numpy array
        y_true = np.hstack([advantages, predictions, actions])
        
        # training Actor and Critic networks
        a_loss = self.Actor.fit(states, y_true, epochs=self.epochs )
        c_loss = self.Critic.fit(states, target, epochs=self.epochs )

        # self.writer.add_scalar('Data/actor_loss_per_replay', np.sum(a_loss.history['loss']), self.replay_count)
        # self.writer.add_scalar('Data/critic_loss_per_replay', np.sum(c_loss.history['loss']), self.replay_count)
        # self.replay_count += 1


    def act(self, state):
        # state = torch.tensor(state.flatten(), dtype=torch.float32).unsqueeze(0)

        # Use the network to predict the next action to take, using the model
        # prediction = self.Actor.forward(np.expand_dims(state, axis=0))[0]
        prediction = self.Actor.forward(state)
        # print("Prediction:", prediction)
        # print("Prediction.item():", prediction.item())
        # print("Type of prediction.item():", type(prediction.item()))
        prediction = prediction.detach().numpy().flatten()
        # print("Numpy prediction:", prediction)
        action = np.random.choice(self.action_space, p=prediction)
        return action, prediction

    def save(self, name="Crypto_trader"):
        # save keras model weights
        self.Actor.save_weights(f"{name}_Actor.pth")
        self.Critic.save_weights(f"{name}_Critic.pth")

    def load(self, name="Crypto_trader"):
        # load keras model weights
        self.Actor.load_weights(f"{name}_Actor.pth")
        self.Critic.load_weights(f"{name}_Critic.pth")
