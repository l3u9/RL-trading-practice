import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np
from utils import *

class DDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DDQN, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(state_size, 128),
            nn.ReLU(),  # Add ReLU activation
            nn.Linear(128, 64),
            nn.ReLU(),  # Add ReLU activation
            nn.Linear(64, action_size)
        )


    def forward(self, x):
        # print(x.shape)
        x = self.model(x)
        return x

class DDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DDQN(state_size, action_size).to(self.device)
        self.target_net = DDQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        act_values = self.policy_net(state_tensor)
        return np.argmax(act_values.detach().cpu().numpy())

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(self.device)
            target = reward
            if not done:
                target = (reward + self.gamma * torch.max(self.target_net(next_state)).item())
            target_f = self.policy_net(state)
            target_f[0, action] = target  # Updated line
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(self.policy_net(state), target_f)
            loss.backward(retain_graph=True)
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


def train_ddqn(env, agent, episodes, batch_size):
    steps = 500
    total_average = deque(maxlen=1000)
    position = ""
    done = False
    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        # for _ in range(steps):
        while True:
            action = agent.act(state)
            next_state, reward, done, action, Date, action = env.step(action)
            total_reward += reward
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if action == 0:
                position = "hold"
            elif action == 1:
                position = "buy"
            elif action == 2:
                position = "sell"
            else:
                position = "error"

            Write_to_file(Date, [env.net_worth, env.crypto_held, reward, env.balance, position])

            if done:
                break
            if env.current_step == env.end_step:
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size) 

        total_average.append(env.net_worth)
        average = np.average(total_average)
        agent.update_target_model()
        print("net worth {} {:.2f} {:.2f} {}".format(e, env.net_worth, average, env.episode_orders))

        # print(f"Episode: {e}/{episodes}, Total reward: {total_reward}, Epsilon: {agent.epsilon}")
