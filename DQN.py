import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DDQN:
    def __init__(self, env, state_size, action_size, hidden_size,
                 learning_rate, batch_size, gamma, epsilon_init, epsilon_decay, epsilon_min, memory_capacity):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_init
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory_capacity = memory_capacity
        self.memory = deque(maxlen=memory_capacity)

        # Q networks
        self.q1_network = QNetwork(state_size, action_size, hidden_size).to(self.device)
        self.q2_network = QNetwork(state_size, action_size, hidden_size).to(self.device)
        self.q2_network.load_state_dict(self.q1_network.state_dict())
        self.optimizer = optim.Adam(self.q1_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return self.choose_best_action(state)

    def choose_best_action(self, state):
        state_tensor = torch.Tensor(state).to(self.device)
        with torch.no_grad():
            return torch.argmax(self.q1_network(state_tensor)).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        if len(self.memory) < self.batch_size: return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_tensor = torch.Tensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).view(-1, 1).to(self.device)
        rewards_tensor = torch.Tensor(rewards).view(-1, 1).to(self.device)
        next_states_tensor = torch.Tensor(next_states).to(self.device)
        dones_tensor = torch.BoolTensor(dones).to(self.device)

        with torch.no_grad():
            actions_next = torch.argmax(self.q1_network(next_states_tensor), 1, keepdims=True)
            q2_values_next = torch.gather(self.q2_network(next_states_tensor), 1, actions_next)

        q1_values = self.q1_network(states_tensor)
        q1_values_selected = torch.gather(q1_values, 1, actions_tensor)
        q1_target = rewards_tensor + self.gamma * q2_values_next * ~dones_tensor

        loss = self.criterion(q1_values_selected, q1_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)

    def update_target_net(self):
        self.q2_network.load_state_dict(self.q1_network.state_dict())

def train_ddqn(agent, env, episodes, timesteps, update_frequency):
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        for t in range(timesteps):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
            total_reward += reward

            if done:
                break

        if episode % update_frequency == 0:
            agent.update_target_net()

        print(f"Episode: {episode}, Total reward: {total_reward}, Epsilon: {agent.epsilon}")
