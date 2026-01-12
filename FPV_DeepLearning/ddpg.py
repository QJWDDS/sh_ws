import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from config import cfg


class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mean - self.x_prev) * self.dt + \
            self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x_initial if self.x_initial is not None else np.zeros_like(self.mean)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, action_dim)
        self.max_action = torch.FloatTensor(max_action).to(cfg.DEVICE)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        return self.max_action * self.tanh(self.l3(x))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, x, u):
        x = torch.cat([x, u], 1)
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        return self.l3(x)


class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(cfg.DEVICE)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(cfg.DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.LR_ACTOR)

        self.critic = Critic(state_dim, action_dim).to(cfg.DEVICE)
        self.critic_target = Critic(state_dim, action_dim).to(cfg.DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.LR_CRITIC)

        self.max_action = max_action
        self.memory = ReplayBuffer(cfg.MEMORY_CAPACITY)
        self.mse_loss = nn.MSELoss()

    def select_action(self, state, noise=None):
        state = torch.FloatTensor(state.reshape(1, -1)).to(cfg.DEVICE)
        action = self.actor(state).cpu().data.numpy().flatten()
        if noise is not None:
            action += noise
        return np.clip(action, -self.max_action, self.max_action)

    def update(self):
        if len(self.memory) < cfg.BATCH_SIZE: return

        state, action, reward, next_state, done = self.memory.sample(cfg.BATCH_SIZE)

        state = torch.FloatTensor(state).to(cfg.DEVICE)
        action = torch.FloatTensor(action).to(cfg.DEVICE)
        reward = torch.FloatTensor(reward).reshape((cfg.BATCH_SIZE, 1)).to(cfg.DEVICE)
        next_state = torch.FloatTensor(next_state).to(cfg.DEVICE)
        done = torch.FloatTensor(done).reshape((cfg.BATCH_SIZE, 1)).to(cfg.DEVICE)

        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_Q = self.critic_target(next_state, next_action)
            target_Q = reward + ((1 - done) * cfg.GAMMA * target_Q)

        current_Q = self.critic(state, action)
        critic_loss = self.mse_loss(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(cfg.TAU * param.data + (1 - cfg.TAU) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(cfg.TAU * param.data + (1 - cfg.TAU) * target_param.data)