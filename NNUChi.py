import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import wandb


class Buffer:
    def __init__(self, n_samples, state_dim, action_dim, gamma=1):
        self.n_samples = n_samples
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.states = None
        self.actions = None
        self.rewards = None
        self.dones = None
        self.ep_start = None
        self.ep_end = None
        self.clear()

    def clear(self):
        self.states = np.zeros((self.n_samples, *self.state_dim))
        self.actions = np.zeros((self.n_samples, *self.action_dim))
        self.rewards = np.zeros((self.n_samples, 1))
        self.dones = np.zeros((self.n_samples, 1)).astype(np.bool)
        self.ep_start = 0
        self.ep_end = 0

    def calculate_reward_to_go(self):
        """turn last episode rewards into reward-to-go"""
        if self.ep_start == self.ep_end:
            return
        for i in range(self.ep_end-1, self.ep_start-1, -1):
            self.rewards[i] += self.gamma * self.rewards[i+1]

    def end_episode(self):
        self.calculate_reward_to_go()
        self.ep_start = self.ep_end

    def add(self, state, action, reward, done):
        self.states[self.ep_end] = state
        self.actions[self.ep_end] = action
        self.rewards[self.ep_end] = reward
        self.dones[self.ep_end] = done
        self.ep_end += 1
        if self.ep_end == self.n_samples:
            self.ep_end = 0
        if done:
            self.end_episode()

    def sample(self, batch_size):
        idx = np.random.randint(0, self.ep_end, min(batch_size, self.ep_end))
        return self.states[idx], self.states[idx+1], self.actions[idx], self.rewards[idx], self.dones[idx]


class NNUChi(nn.Module):
    def __init__(self, env, beta=1, hidden_dim=32, use_wandb=True, u_ref_state=None):
        super().__init__()
        self.env = env
        self.beta = beta
        self.state_dim = env.observation_space.n
        self.action_dim = env.action_space.n
        self.hidden_dim = hidden_dim

        self.u1 = nn.Linear(self.state_dim + self.action_dim, hidden_dim)
        self.u2 = nn.Linear(hidden_dim, hidden_dim)
        self.u3 = nn.Linear(hidden_dim, 1)

        self.alpha = 1e-2
        self.logu_optimizer = Adam(self.parameters(), lr=self.alpha)

        self.use_wandb = use_wandb
        self.device = None
        self.exploration = 0.9
        self.exploration_decay = 0.99995

        self.chi_ref_state = None
        self.u_ref_state = None
        self.reference_reward = None
        self.ref_chi = None
        self._set_references(u_ref_state=u_ref_state)

    def _set_references(self, u_ref_state=None):
        if u_ref_state is None:
            self.u_ref_state = (0, 0)
        else:
            self.u_ref_state = u_ref_state

        self.env.reset()
        _, action = self.u_ref_state
        self.chi_ref_state, self.reference_reward, _, _ = self.env.step(action)
        self.ref_chi = 0.5  # ref_chi = self.init_chi[self.chi_ref_state]
        self.env.reset()
        sa = np.array([0])
        self.ref_sa = th.tensor(sa, dtype=th.float32, device=self.device)

    def logu_forward(self, x):
        x = F.relu(self.u1(x))
        x = F.relu(self.u2(x))
        x = F.tanh(self.u3(x))
        return x

    def get_state_actions(self, x):
        # extend state dimension to include actions,
        x = np.concatenate([np.zeros(((x.shape[0],) if len(
            x.shape) > 1 else ()) + (self.action_dim,)), x], axis=-1)
        # and add a dimension for a specific one-hot action
        # the dimension should be (action, [batch], state+action)
        actions = np.eye(self.action_dim)
        # extend actions to include states in the first dimension
        actions = np.concatenate(
            [np.zeros((self.action_dim, self.state_dim)), actions], axis=-1)
        # add a batch dimension if its a batch
        if len(x.shape) > 1:
            actions = actions.reshape((actions.shape[0], 1, actions.shape[1]))
        x = x.reshape((1,) + x.shape)
        x = x + actions
        return x

    def action(self, x: np.array, deterministic=False) -> int:
        if np.random.rand() < self.exploration and not deterministic:
            return self.env.action_space.sample()
        x = self.get_state_actions(x)
        x = th.tensor(x, dtype=th.float32, device=self.device)
        # Grab a random sample according to the distribution
        # pi(a|s) = u(s,a) / sum_a u(s,a)
        u_sa = th.exp(self.logu_forward(x))
        u_s = u_sa.sum(dim=0)
        pi_sa = u_sa / u_s
        # Sample an action according to the distribution
        # a = np.random.choice(
        #     self.action_dim, p=pi_sa.cpu().detach().numpy().flatten())
        # Grab the action with the highest probability
        a = th.argmax(pi_sa).item()
        return a

    def set_device(self, device):
        self.device = device
        self.to(device)

    def update(self, buffer, batch_size):

        states, next_states, actions, rewards, dones = buffer.sample(
            batch_size)
        state_actions = np.concatenate([states, actions], axis=1)
        sa_next = self.get_state_actions(next_states)
        sa_next_tensor = th.tensor(
            sa_next, dtype=th.float32, device=self.device)

        state_actions = th.tensor(
            state_actions, dtype=th.float32, device=self.device)
        rewards = th.tensor(rewards, dtype=th.float32, device=self.device)
        # next_states_tensor = th.tensor(
        #     next_states, dtype=th.float32, device=self.device)
        logu_sa = self.logu_forward(state_actions)
        logu_sa_prime = self.logu_forward(sa_next_tensor)
        # chi_sp = self.chi_forward(next_states_tensor)
        chi_sp = th.exp(logu_sa_prime).sum(dim=1, keepdim=True)

        delta_rewards = rewards - self.reference_reward
        target_logu_values = (
            self.beta * delta_rewards + th.log(chi_sp/self.ref_chi)) - 1
        # target_logu_values = target_logu_values - \
        #     self.logu_forward(self.ref_sa)

        logu_loss = F.mse_loss(logu_sa, target_logu_values)
        self.logu_optimizer.zero_grad()
        logu_loss.backward()
        self.logu_optimizer.step()

    def learn(self, total_timesteps: int = 1_000_000, rollout_len: int = 1_000, batch_size: int = 10_000, epochs: int = 10):
        """
        :param n_total_steps: total number of steps to train for
        :param rollout_len: number of steps to collect before updating the policy
        :param batch_size: number of samples to use for each update
        :param epochs: number of epochs to train for
        """
        buffer = Buffer(int(1e6), (self.state_dim,), (self.action_dim,))
        state = self.env.reset()
        mean_rollout_reward = 0
        self.set_device("cpu")  # for faster environment interaction
        for i in range(1, total_timesteps+1):
            state_one_hot = np.zeros(self.state_dim)
            state_one_hot[state] = 1
            action = self.action(state_one_hot)
            action_one_hot = np.zeros(self.action_dim)
            action_one_hot[action] = 1
            next_state, reward, done, _ = self.env.step(action)
            buffer.add(state_one_hot, action_one_hot, reward, done)
            state = next_state
            mean_rollout_reward += reward
            self.exploration = max(
                0.05, self.exploration*self.exploration_decay)
            if done:
                self.env.reset()
            if i % rollout_len == 0:
                self.set_device("cuda" if th.cuda.is_available() else "cpu")
                # log rollout mean reward
                mean_rollout_reward /= rollout_len
                if self.use_wandb:
                    wandb.log({
                        "mean_rollout_reward": mean_rollout_reward,
                        "step": i,
                        "exploration": self.exploration,
                    })
                mean_rollout_reward = 0
                buffer.end_episode()
                for _ in range(epochs):
                    self.update(buffer, batch_size)
