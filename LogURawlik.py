import gym
import torch
import torch.nn as nn
from collections import deque
import random
from torch.distributions import Categorical
import numpy as np
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.vec_env import unwrap_vec_normalize
from stable_baselines3.common.logger import configure
import time
from torch.nn import functional as F
from stable_baselines3.common.utils import polyak_update

from utils import logger_at_folder
from ReplayBuffers import Memory

class LogUNet(nn.Module):
    def __init__(self, env, device='cuda', hidden_dim=128):
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
    
    def get_chi(self, logu_a, prior_a=None):
        if prior_a is None:
            prior_a = 1 / self.nA
        chi = torch.sum(prior_a * torch.exp(logu_a))
        return chi
        
    def choose_action(self, state, greedy=False, prior_a=None):
        with torch.no_grad():
            if prior_a is None:
                prior_a = 1 / self.nA
            logu = self.forward(state)

            if greedy:
                # raise NotImplementedError("Greedy not implemented (possible bug?).")
                a = logu.argmax()
                return a.item()

            # First subtract a baseline:
            logu = logu - (torch.max(logu) + torch.min(logu))/2
            dist = torch.exp(logu) * prior_a
            dist = dist / torch.sum(dist)
            c = Categorical(dist)
            a = c.sample()

        return a.item()



class LogULearner:
    def __init__(self, 
                 env_id,
                 beta,
                 learning_rate,
                 batch_size,
                 buffer_size,
                 target_update_interval,
                 tau,
                 hidden_dim=64,
                 tau_theta=0.001,
                 gradient_steps=1,
                 train_freq=-1,
                 max_grad_norm=10,
                 prior_update_interval=-1,
                 device='cpu',
                 log_dir=None,
                 log_interval=1000,
                 save_checkpoints = False,
                 ) -> None:
        self.env = gym.make(env_id)
        # make another instance for evaluation purposes only:
        self.eval_env = gym.make(env_id)
        # self._vec_normalize_env = unwrap_vec_normalize(self.env)
        self.beta = beta
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.tau = tau
        self.hidden_dim = hidden_dim
        self.gradient_steps = gradient_steps
        self.device = device
        self.save_checkpoints = save_checkpoints
        self.log_interval = log_interval
        self.tau_theta = tau_theta
        self.train_freq = train_freq
        self.max_grad_norm = max_grad_norm
        self.prior_update_interval = prior_update_interval
        self.prior = None
        
        self.replay_buffer = Memory(buffer_size, device=device)
        self.ref_action = None
        self.ref_state = None
        self.ref_reward = None
        self.theta = torch.Tensor([0]).to(self.device)
        self.eval_auc = 0
        self.num_episodes = 0
        # self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Set up the logger:
        self.logger = logger_at_folder(log_dir, algo_name='Rawlik')

        self._n_updates = 0
        self.env_steps = 0
        self._initialize_networks()

    def _initialize_networks(self):

        self.online_logu = LogUNet(self.env, device=self.device)
        self.target_logu = LogUNet(self.env, device=self.device)
        self.target_logu.load_state_dict(self.online_logu.state_dict())

        # Make LogU learnable:
        self.optimizer = torch.optim.Adam(self.online_logu.parameters(), lr=self.learning_rate)


    def train(self,):
        # replay = self.replay_buffer.sample(self.batch_size, env=self._vec_normalize_env)
        # average self.theta over multiple gradient steps
        new_thetas = []
        for grad_step in range(self.gradient_steps):
            replay = self.replay_buffer.sample(self.batch_size, continuous=False)
            states, next_states, actions, next_actions, rewards, dones = replay

            actions = actions.unsqueeze(1)
            next_actions = next_actions.unsqueeze(1)
            rewards = rewards.unsqueeze(1)
            dones = dones.unsqueeze(1)

            curr_logu = self.online_logu(states)
            curr_logu = curr_logu.squeeze(1)#self.online_logu(states).squeeze(1)
            curr_logu = curr_logu.gather(1, actions.long())

            with torch.no_grad():
                ref_logu = self.online_logu(self.ref_next_state)
                ref_chi = self.online_logu.get_chi(ref_logu)
                new_theta = self.ref_reward - torch.log(ref_chi)
                new_thetas.append(new_theta)

                target_next_logu = self.target_logu(next_states)
                next_logu = target_next_logu.gather(1, next_actions.long())
                
                expected_curr_logu = self.beta * (rewards + self.theta) + (1 - dones) * next_logu

                
            self.logger.record("theta", self.theta.item())
            self.logger.record("avg logu", curr_logu.mean().item())
            # Huber loss:
            loss = F.smooth_l1_loss(curr_logu, expected_curr_logu)
            # MSE loss:
            # loss = F.mse_loss(curr_logu, expected_curr_logu)
            self.logger.record("loss", loss.item())
            self.optimizer.zero_grad()
            # Increase update counter
            self._n_updates += self.gradient_steps
            
            # Clip gradient norm
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.online_logu.parameters(), self.max_grad_norm)

            # Log the average gradient:
            # TODO: put this in a parallel process somehow or use dot prods?
            total_norm = torch.max(torch.stack(
                        [p.grad.detach().abs().max() for p in self.online_logu.parameters()]
                        ))
            self.logger.record("max_grad", total_norm.item())
            self.optimizer.step()
        # self.theta = torch.mean(torch.stack(new_thetas))
        new_thetas = torch.stack(new_thetas)
        # new_thetas = torch.clamp(new_thetas, 0, -1)

        self.theta = self.tau_theta*self.theta + (1 - self.tau_theta) * torch.mean(new_thetas)
    
    def learn(self, total_timesteps):
        # Start a timer to log fps:
        t0 = time.thread_time_ns()

        while self.env_steps < total_timesteps:
            state = self.env.reset()
            if self.env_steps == 0:
                self.ref_state = state
            episode_reward = 0
            done = False
            action = self.online_logu.choose_action(state)

            self.num_episodes += 1
            self.rollout_reward = 0
            while not done:
                torch_state = torch.FloatTensor(state).to(self.device)
                action = self.online_logu.choose_action(torch_state, prior_a=self.prior)
                next_state, reward, done, _ = self.env.step(action)
                self.rollout_reward += reward
                if self.env_steps == 0:
                    self.ref_action = action
                    self.ref_reward = reward
                    self.ref_next_state = next_state

                #TODO: Shorten this: (?)
                if (self.train_freq == -1 and done) or (self.train_freq != -1 and self.env_steps % self.train_freq == 0):
                    if self.replay_buffer.size() > self.batch_size: # or learning_starts?
                        self.train()


                if self.env_steps % self.target_update_interval == 0:
                    # Do a Polyak update of parameters:
                    polyak_update(self.online_logu.parameters(), self.target_logu.parameters(), self.tau)


                if self.prior_update_interval != -1 and self.env_steps % self.prior_update_interval == 0:
                    # Update pi_0:
                    next_target_logu = self.target_logu(next_state)
                    next_target_logu = next_target_logu.squeeze(0)
                    self.next_prior = torch.exp(next_target_logu) / torch.sum(torch.exp(next_target_logu))
                    target_logu = self.target_logu(state)
                    target_logu = target_logu.squeeze(0)
                    self.prior = torch.exp(target_logu) / torch.sum(torch.exp(target_logu))

                
                self.env_steps += 1
                next_action = self.online_logu.choose_action(next_state, prior_a=self.prior)
                episode_reward += reward
                self.replay_buffer.add((state, next_state, action, next_action, reward, done))
                state = next_state
                action = next_action
                
                if self.env_steps % self.log_interval == 0:
                    # end timer:
                    t_final = time.thread_time_ns()
                    # fps averaged over log_interval steps:
                    fps = self.log_interval / ((t_final - t0) / 1e9)

                    avg_eval_rwd = self.evaluate()
                    self.eval_auc += avg_eval_rwd
                    if self.save_checkpoints:
                        torch.save(self.online_logu.state_dict(), 'sql-policy.para')
                    self.logger.record("env. steps", self.env_steps)
                    self.logger.record("eval/avg_reward", avg_eval_rwd)
                    self.logger.record("eval/auc", self.eval_auc)
                    self.logger.record("num. episodes", self.num_episodes)
                    self.logger.record("fps", fps)
                    self.logger.dump(step=self.env_steps)
                    t0 = time.thread_time_ns()
                self.logger.record("Rollout reward:", self.rollout_reward)


    def evaluate(self, n_episodes=1):
        # run the current policy and return the average reward
        avg_reward = 0.
        # Wrap a timelimit:
        # self.eval_env = TimeLimit(self.eval_env, max_episode_steps=500)
        # move to cpu for evaluation:
        # self.eval_env = self.eval_env.to('cpu')
        for ep in range(n_episodes):
            state = self.eval_env.reset()
            done = False
            while not done:
                action = self.online_logu.choose_action(state, greedy=True)
                # if ep == 0:
                # self.env.render()

                next_state, reward, done, _ = self.eval_env.step(action)

                avg_reward += reward
                state = next_state
        avg_reward /= n_episodes
        self.eval_env.close()
        return avg_reward
    
def main():
    env = 'CartPole-v1'
    env = 'LunarLander-v2'
    from hparams import cartpole_rawlik as config
    agent = LogULearner(env, **config, log_interval=1500)
    agent.learn(5000_000)


if __name__ == '__main__':
    for _ in range(1):
        main()
