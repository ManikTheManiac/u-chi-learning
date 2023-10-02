import gym
import torch
import torch.nn as nn
from torch.distributions import Categorical
from stable_baselines3.common.preprocessing import preprocess_obs
import numpy as np
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
from stable_baselines3.sac.policies import Actor
from stable_baselines3.common.utils import polyak_update

class LogUNet(nn.Module):
    def __init__(self, env, device='cuda', hidden_dim=256):
        super(LogUNet, self).__init__()
        self.env = env
        self.device = device
        try:
            self.nS = env.observation_space.n
        except AttributeError:
            self.nS = env.observation_space.shape[0]

        self.nA = env.action_space.n
        self.fc1 = nn.Linear(self.nS, hidden_dim, device=self.device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, device=self.device)
        self.fc3 = nn.Linear(hidden_dim, self.nA, device=self.device)
        self.relu = nn.ReLU()
     
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32).detach()  # Convert to PyTorch tensor

        x = preprocess_obs(x, self.env.observation_space)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x

    def get_chi(self, logu_a, prior=None):
        if prior is None:
            prior = 1 / self.nA

        chi = torch.sum(prior * torch.exp(logu_a))
        return chi
        
    def choose_action(self, state, greedy=False, prior=None):
        if prior is None:
            prior = 1 / self.nA
        with torch.no_grad():
            logu = self.forward(state)

            if greedy:
                a = (logu * prior).argmax()
                return a.item()

            # First subtract a baseline:
            logu = logu - (torch.max(logu) + torch.min(logu))/2
            dist = torch.exp(logu) * prior
            dist = dist / torch.sum(dist)
            c = Categorical(dist)
            a = c.sample()

        return a.item()


class UNet(nn.Module):
    def __init__(self, env, device='cuda', hidden_dim=256):
        super(UNet, self).__init__()
        self.env = env
        self.device = device
        try:
            self.nS = env.observation_space.n
        except AttributeError:
            self.nS = env.observation_space.shape[0]

        self.nA = env.action_space.n
        self.fc1 = nn.Linear(self.nS, hidden_dim, device=self.device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, device=self.device)
        self.fc3 = nn.Linear(hidden_dim, self.nA, device=self.device)
        self.relu = nn.Tanh()
        # self.relu = nn.LeakyReLU()
     
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32).detach()  # Convert to PyTorch tensor

        x = preprocess_obs(x, self.env.observation_space)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return torch.abs(x)
        # return self.relu(x+4) + 1e-6

    def get_chi(self, u_a):
        prior_policy = 1 / self.nA
        chi = torch.sum(prior_policy * u_a)
        return chi
        
    def choose_action(self, state, greedy=False):
        with torch.no_grad():
            u = self.forward(state)

            if greedy:
                a = u.argmax()
                return a.item()

            # First subtract a baseline:
            # u = u - (torch.max(u) + torch.min(u))/2
            # print(u)
            dist = u * 1 / self.nA
            dist = dist / torch.sum(dist)
            # print(dist)
            c = Categorical(dist)
            a = c.sample()

        return a.item()


class Optimizers():
    def __init__(self, list_of_optimizers: list):
        self.optimizers = list_of_optimizers

    def zero_grad(self):
        for opt in self.optimizers:
            opt.zero_grad()

    def step(self):
        for opt in self.optimizers:
            opt.step()


class TargetNets():
    def __init__(self, list_of_nets):
        self.nets = list_of_nets

    def __iter__(self):
        return iter(self.nets)

    def load_state_dict(self, list_of_state_dicts):
        for new_state, net in zip(list_of_state_dicts, self.nets):
            net.load_state_dict(new_state)

    def polyak(self, new_weights, tau):
        for weights, new_weights in zip(self.parameters(), new_weights):
            polyak_update(new_weights, weights, tau)

    def parameters(self):
        return [net.parameters() for net in self.nets]


class OnlineNets():
    def __init__(self, list_of_nets):
        self.nets = list_of_nets

    def __iter__(self):
        return iter(self.nets)

    def greedy_action(self, state):
        with torch.no_grad():
            logu = torch.stack([net(state) for net in self.nets])
            logu = logu.squeeze(1)
            logu = torch.min(logu, dim=0)[0]
            greedy_action = logu.argmax()
        return greedy_action.item()

    def choose_action(self, state):
        # Get a sample from each net, then sample uniformly over them:
        actions = [net.choose_action(state) for net in self.nets]
        action = np.random.choice(actions)
        # perhaps re-weight this based on pessimism?
        return action.item()

    def parameters(self):
        return [net.parameters() for net in self.nets]

    def clip_grad_norm(self, max_grad_norm):
        for net in self.nets:
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)


