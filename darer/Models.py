import torch
import torch.nn as nn
from torch.distributions import Categorical
from stable_baselines3.common.preprocessing import preprocess_obs, get_action_dim, get_flattened_obs_dim, get_obs_shape
import numpy as np
from stable_baselines3.common.utils import zip_strict
from gymnasium import spaces
import gymnasium as gym

class LogUNet(nn.Module):
    def __init__(self, env, device='cuda', hidden_dim=256):
        super(LogUNet, self).__init__()
        self.env = env
        self.device = device
        if isinstance(env.observation_space, spaces.Discrete):
            self.nS = env.observation_space.n
        elif isinstance(env.observation_space, spaces.Box):
            self.nS = env.observation_space.shape[0]       
        self.nA = env.action_space.n
        self.fc1 = nn.Linear(self.nS, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, self.nA)
        self.relu = nn.ReLU()
        self.to(device)
     
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device)#, dtype=torch.float32).detach()  # Convert to PyTorch tensor

        x = preprocess_obs(x, self.env.observation_space)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x
        
    def choose_action(self, state, greedy=False, prior=None):
        if prior is None:
            prior = 1 / self.nA
        with torch.no_grad():
            logu = self.forward(state)

            if greedy:
                # not worth exponentiating since it is monotonic
                a = (logu * prior).argmax(dim=-1)
                return a.item()

            # First subtract a baseline:
            logu = logu - (torch.max(logu) + torch.min(logu))/2
            dist = torch.exp(logu) * prior
            # dist = dist / torch.sum(dist)
            c = Categorical(dist)
            a = c.sample()

        return a.item()


class UNet(nn.Module):
    def __init__(self, env, device='cuda', hidden_dim=256):
        super(UNet, self).__init__()
        self.env = env
        self.device = device
        if isinstance(env.observation_space, spaces.Discrete):
            self.nS = env.observation_space.n
        elif isinstance(env.observation_space, spaces.Box):
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
    def __len__(self):
        return len(self.nets)

    def __iter__(self):
        return iter(self.nets)

    def load_state_dicts(self, list_of_state_dicts):
        """
        Load state dictionaries into target networks.

        Args:
            list_of_state_dicts (list): A list of state dictionaries to load into the target networks.

        Raises:
            ValueError: If the number of state dictionaries does not match the number of target networks.
        """
        if len(list_of_state_dicts) != len(self):
            raise ValueError("Number of state dictionaries does not match the number of target networks.")
        
        for new_state, target_net in zip(list_of_state_dicts, self):
            target_net.load_state_dict(new_state)

    def polyak(self, online_nets, tau):
        """
        Perform a Polyak (exponential moving average) update for target networks.

        Args:
            online_nets (list): A list of online networks whose parameters will be used for the update.
            tau (float): The update rate, typically between 0 and 1.

        Raises:
            ValueError: If the number of online networks does not match the number of target networks.
        """
        if len(online_nets) != len(self.nets):
            raise ValueError("Number of online networks does not match the number of target networks.")

        with torch.no_grad():
            for online_net, target_net in zip(online_nets, self.nets):
                for online_param, target_param in zip(online_net.parameters(), target_net.parameters()):
                    target_param.data.mul_(tau).add_(online_param.data, alpha=1.0 - tau)

    def parameters(self):
        """
        Get the parameters of all target networks.

        Returns:
            list: A list of network parameters for each target network.
        """
        return [net.parameters() for net in self.nets]


import torch
import torch.distributions as dist

class OnlineNets:
    """
    A utility class for managing online networks in reinforcement learning.

    Args:
        list_of_nets (list): A list of online networks.
    """

    def __init__(self, list_of_nets):
        self.nets = list_of_nets

    def __len__(self):
        return len(self.nets)

    def __iter__(self):
        return iter(self.nets)

    def greedy_action(self, state):
        """
        Select a greedy action based on the online networks.

        Args:
            state (torch.Tensor): The input state.

        Returns:
            int: The index of the greedy action.
        """
        with torch.no_grad():
            logu_stacked = torch.stack([net(state) for net in self])
            logu = logu_stacked.squeeze(1)
            logu = torch.min(logu, dim=0)[0]
            greedy_action = logu.argmax()
            # check if the nets agree:
            all_greedys = logu_stacked.argmax(dim=-1)
            agreed = torch.all(all_greedys == greedy_action)
            # print(agreed)
        return greedy_action.item(), agreed

    def choose_action(self, state, prior=None):
        # Validate the input state
        # assert isinstance(state, torch.Tensor), "Input state must be a PyTorch tensor"
        # assert state.shape == (batch_size, state_dim), "Invalid state shape"

        # Get actions from each network
        logus = torch.stack([net.forward(state) for net in self])
        logus = logus.squeeze(1)
        logu = torch.min(logus, dim=0)[0] # pessimistic values for approximation of true value
        
        # print(actions)
        if prior is None:
            # If no prior is provided, sample uniformly
            # action = np.random.choice(actions)
            
            # First subtract a baseline:
            logu = logu - (torch.max(logu) + torch.min(logu))/2
            # print(u)
            u = torch.exp(logu)
            dist = u * 1 / 2
            dist = dist / torch.sum(dist)
            # print(dist)
            c = Categorical(dist)
            action = c.sample()

            # action = actions[0]
        # else:
        #     # Ensure that prior is a list of weights (e.g., [0.2, 0.3, 0.5])
        #     assert isinstance(prior, list), "Prior must be a list of weights"
        #     assert len(prior) == len(actions), "Prior length must match the number of networks"

        #     # Normalize the prior weights to create a probability distribution
        #     prior_prob = [weight / sum(prior) for weight in prior]

        #     # Sample an action based on the prior distribution
        #     action = np.random.choice(actions, p=prior_prob)

        return action.item()

    def parameters(self):
        """
        Get the parameters of all online networks.

        Returns:
            list: A list of network parameters for each online network.
        """
        return [net.parameters() for net in self]

    def clip_grad_norm(self, max_grad_norm):
        """
        Clip gradients for all online networks.

        Args:
            max_grad_norm (float): Maximum gradient norm for clipping.
        """
        for net in self:
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)

class LogUsa(nn.Module):
    def __init__(self, env, hidden_dim=256, device='cuda'):
        super(LogUsa, self).__init__()
        self.env = env
        self.device = device
        self.nS = get_flattened_obs_dim(self.env.observation_space)
        self.nA = get_action_dim(self.env.action_space)
        self.fc1 = nn.Linear(self.nS + self.nA, hidden_dim, device=self.device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, device=self.device)
        self.fc3 = nn.Linear(hidden_dim, 1, device=self.device)
        self.relu = nn.ReLU()

    def forward(self, obs, action):
        obs = torch.Tensor(obs).to(self.device)
        action = torch.Tensor(action).to(self.device)
        obs = preprocess_obs(obs, self.env.observation_space)
        x = torch.cat([obs, action], dim=-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 5
LOG_SIG_MIN = -30
epsilon = 1e-6


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class GaussianPolicy(nn.Module):
    def __init__(self, hidden_dim, observation_space, action_space, use_action_bounds=False, device='cpu'):
        super(GaussianPolicy, self).__init__()
        self.device = device
        num_inputs = get_flattened_obs_dim(observation_space)
        num_actions = get_action_dim(action_space)
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)#, device=self.device)
            self.action_bias = torch.tensor(0.)#, device=self.device)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)#, device=self.device)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)#, device=self.device)
            
        self.observation_space = observation_space
        self.to(device)

    def forward(self, obs):
        obs = torch.Tensor(obs).to(self.device)
        obs = preprocess_obs(obs, self.observation_space)
        x = F.relu(self.linear1(obs))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        # print(std)
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        noisy_action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return noisy_action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)
    
