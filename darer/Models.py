import torch
import torch.nn as nn
from torch.distributions import Categorical
from stable_baselines3.common.preprocessing import is_image_space, preprocess_obs, get_action_dim, get_flattened_obs_dim, get_obs_shape
import numpy as np
from stable_baselines3.common.utils import zip_strict
from gymnasium import spaces
import gymnasium as gym

class LogUNet(nn.Module):
    def __init__(self, env, device='cuda', hidden_dim=256, activation=nn.ReLU):
        super(LogUNet, self).__init__()
        self.env = env
        self.nA = env.action_space.n
        self.is_image_space = is_image_space(env.observation_space)

        self.device = device
        if isinstance(env.observation_space, spaces.Discrete):
            self.nS = env.observation_space.n
        elif isinstance(env.observation_space, spaces.Box):
            # check if image:
            if is_image_space(env.observation_space):
                self.nS = get_flattened_obs_dim(env.observation_space)
                # Use a CNN:
                n_channels = env.observation_space.shape[2]
                model = nn.Sequential(
                    nn.Conv2d(n_channels, 64, kernel_size=8, stride=4),
                    activation(),
                    nn.Conv2d(64, 32, kernel_size=4, stride=2),
                    activation(),
                    nn.Conv2d(32, 32, kernel_size=3, stride=1),
                    activation(),
                    nn.Flatten(start_dim=1, end_dim=-1),
                )
                model.to(self.device)
                # calculate resulting shape for FC layers:
                rand_inp = env.observation_space.sample()
                x = torch.tensor(rand_inp, device=self.device, dtype=torch.float32)  # Convert to PyTorch tensor
                x = x.detach()
                x = preprocess_obs(x, self.env.observation_space)
                x = x.permute([2,0,1]).unsqueeze(0)
                flat_size = model(x).shape[1]
                print(f"Using a CNN with {flat_size}-dim. outputs.")
                # flat part
                model.extend(nn.Sequential(
                    nn.Linear(flat_size, hidden_dim),
                    activation(),
                    nn.Linear(hidden_dim, hidden_dim),
                    activation(),
                    nn.Linear(hidden_dim, self.nA),
                ))
            else:
                self.nS = env.observation_space.shape[0]

                model = (nn.Sequential(
                    nn.Linear(self.nS, hidden_dim),
                    activation(),
                    nn.Linear(hidden_dim, hidden_dim),
                    activation(),
                    nn.Linear(hidden_dim, self.nA),
                ))
            self.model = model
        self.to(device)
     
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)  # Convert to PyTorch tensor
        # x = x.detach()
        x = preprocess_obs(x, self.env.observation_space)
        # Reshape the image:
        if self.is_image_space:
            if len(x.shape) == 3:
                # Single image
                x = x.permute([2,0,1])
                x = x.unsqueeze(0)
            else:
                # Batch of images
                x = x.permute([0,3,1,2])
        x = self.model(x)
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
            # clamp to avoid overflow:
            logu = torch.clamp(logu, min=-20, max=20)
            dist = torch.exp(logu) * prior
            # dist = dist / torch.sum(dist)
            c = Categorical(dist)#, validate_args=True)
            # c = Categorical(logits=logu*prior)
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
        
        for online_net_dict, target_net in zip(list_of_state_dicts, self):
            
            target_net.load_state_dict(online_net_dict)

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
            # zip does not raise an exception if length of parameters does not match.
            for new_params, target_params in zip(online_nets.parameters(), self.parameters()):
                for new_param, target_param in zip_strict(new_params, target_params):
                    # target_param.data.mul_(tau)
                    # new_param.data.mul_(1 - tau)
                    # target_param.data.add_(new_param.data)
                    target_param.data.mul_(tau).add_(new_param.data, alpha=1.0-tau)
                    # torch.add(target_param.data, new_param.data, out=target_param.data)

    def parameters(self):
        """
        Get the parameters of all target networks.

        Returns:
            list: A list of network parameters for each target network.
        """
        return [net.parameters() for net in self.nets]


class OnlineNets():
    """
    A utility class for managing online networks in reinforcement learning.

    Args:
        list_of_nets (list): A list of online networks.
    """
    def __init__(self, list_of_nets, aggregator='min'):
        self.nets = list_of_nets
        if aggregator == 'min':
            self.aggregator = torch.min
        elif aggregator == 'mean':
            self.aggregator = torch.mean
        elif aggregator == 'max':
            self.aggregator = torch.max

    def __len__(self):
        return len(self.nets)
    
    def __iter__(self):
        return iter(self.nets)
    
    def greedy_action(self, state):
        with torch.no_grad():
            # logu = torch.stack([net(state) for net in self.nets], dim=-1)
            # logu = logu.squeeze(1)
            # logu = self.aggregator(logu, dim=-1)[0]
            
            # greedy_action = logu.argmax()
            # greedy_actions = [net(state).argmax().cpu() for net in self.nets]
            greedy_actions = [net.choose_action(state, greedy=True) for net in self.nets]
            greedy_action = np.random.choice(greedy_actions)
        return greedy_action
        # return greedy_action.item()

    def choose_action(self, state):
        # Get a sample from each net, then sample uniformly over them:
        actions = [net.choose_action(state) for net in self.nets]
        action = np.random.choice(actions)
        # perhaps re-weight this based on pessimism?
        return action

    def parameters(self):
        return [net.parameters() for net in self]

    def clip_grad_norm(self, max_grad_norm):
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
    
