import gymnasium as gym
import torch
import numpy as np
from torch.nn import functional as F
import time
from stable_baselines3.common.buffers import ReplayBuffer
import wandb
from Models import LogUNet, OnlineNets, Optimizers, TargetNets
from utils import env_id_to_envs, logger_at_folder
from stable_baselines3.common.preprocessing import preprocess_obs
# raise warning level for debugger:
import warnings
# warnings.filterwarnings("error")


class LogULearner:
    def __init__(self,
                 env_id,
                 beta,
                 learning_rate,
                 batch_size,
                 buffer_size,
                 target_update_interval,
                 tau,
                 theta_update_interval=1,
                 hidden_dim=64,
                 num_nets=2,
                 tau_theta=0.001,
                 gradient_steps=1,
                 train_freq=-1,
                 max_grad_norm=10,
                 learning_starts=1000,
                 device='cpu',
                 render=False,
                 log_dir=None,
                 log_interval=1000,
                 save_checkpoints=False,
                 use_wandb=False,
                 ) -> None:
        
        self.env, self.eval_env = env_id_to_envs(env_id, render)
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
        self.theta_update_interval = theta_update_interval
        self.train_freq = train_freq
        self.max_grad_norm = max_grad_norm
        self.num_nets = num_nets
        self.prior = None
        self.learning_starts = learning_starts
        self.use_wandb = use_wandb

        # self.replay_buffer = Memory(buffer_size, device=device)
        self.replay_buffer = ReplayBuffer(buffer_size=buffer_size,
                                          observation_space=self.env.observation_space,
                                          action_space=self.env.action_space,
                                          n_envs=1,
                                          handle_timeout_termination=False,
                                          device=device)
        self.ref_action = None
        self.ref_state = None
        self.ref_reward = None
        self.theta = torch.Tensor([0]).to(self.device)
        self.eval_auc = 0
        self.num_episodes = 0

        # Set up the logger:
        self.logger = logger_at_folder(log_dir, algo_name=f'acro1')

        self._n_updates = 0
        self.env_steps = 0
        self._initialize_networks()


    def _initialize_networks(self):
        self.online_logus = OnlineNets(list_of_nets=[LogUNet(self.env, hidden_dim=self.hidden_dim, device=self.device)
                                                     for _ in range(self.num_nets)])
        self.target_logus = TargetNets(list_of_nets=[LogUNet(self.env, hidden_dim=self.hidden_dim, device=self.device)
                                                     for _ in range(self.num_nets)])
        self.target_logus.load_state_dict(
            [logu.state_dict() for logu in self.online_logus])
        # Make (all) LogUs learnable:
        opts = [torch.optim.Adam(logu.parameters(), lr=self.learning_rate)
                for logu in self.online_logus]
        self.optimizers = Optimizers(opts)

    def train(self,):
        # replay = self.replay_buffer.sample(self.batch_size, env=self._vec_normalize_env)
        # average self.theta over multiple gradient steps
        new_thetas = torch.zeros(
            self.gradient_steps, self.num_nets).to(self.device)
        for grad_step in range(self.gradient_steps):
            replay = self.replay_buffer.sample(self.batch_size)
            states, actions, next_states, dones, rewards = replay
            curr_logu = torch.cat([online_logu(states).squeeze().gather(1, actions.long())
                                   for online_logu in self.online_logus], dim=1)
            with torch.no_grad():
                # ref_logu = [logu(self.ref_next_state)
                #             for logu in self.online_logus]
                # # since pi0 is same for all, just do exp(ref_logu) and sum over actions:
                # ref_chi = torch.stack([torch.exp(ref_logu_val).sum(dim=-1) / self.env.action_space.n
                #                        for ref_logu_val in ref_logu], dim=-1)
                # new_thetas[grad_step, :] = self.ref_reward - torch.log(ref_chi)


                # target_next_logu = torch.cat([target_logu(next_states).squeeze().gather(1, next_actions.long())
                #                               for target_logu in self.target_logus], dim=1)
                target_next_logu = torch.stack([target_logu(next_states).sum(dim=-1) / self.env.action_space.n
                                                for target_logu in self.target_logus], dim=1)

                next_logu, _ = torch.min(target_next_logu, dim=1)
                if isinstance(self.env.observation_space, gym.spaces.Discrete):
                    # pass
                    target_next_logu = target_next_logu.squeeze(-1)
                else:
                    next_logu = next_logu.unsqueeze(-1)
                # average over the batch:
                batch_theta_per_net = torch.mean((self.beta * rewards + target_next_logu - curr_logu) / -self.beta, dim=0)
                new_thetas[grad_step, :] = batch_theta_per_net


                expected_curr_logu = self.beta * \
                    (rewards + self.theta) + (1 - dones) * next_logu
                expected_curr_logu = expected_curr_logu.squeeze(1)

            
            self.logger.record("train/theta", self.theta.item())
            self.logger.record("train/avg logu", curr_logu.mean().item())

            # Huber loss:
            loss = 0.5*sum(F.smooth_l1_loss(logu, expected_curr_logu)
                           for logu in curr_logu.T)
            # MSE loss:
            # loss = F.mse_loss(curr_logu, expected_curr_logu)
            self.logger.record("train/loss", loss.item())
            self.optimizers.zero_grad()
            # Increase update counter
            self._n_updates += self.gradient_steps

            # Clip gradient norm
            loss.backward()
            self.online_logus.clip_grad_norm(self.max_grad_norm)

            # Log the average gradient:
            # TODO: put this in a parallel process somehow or use dot prods?
            # total_norm = torch.max(torch.stack(
            #             [p.grad.detach().abs().max() for p in self.online_logu.parameters()]
            #             ))
            # self.logger.record("max_grad", total_norm.item())
            self.optimizers.step()
        # new_thetas = torch.clamp(new_thetas, 0, -1)
        # Log both theta values:
        # average over the gradient steps:
        new_thetas = new_thetas.mean(dim=0)
        for idx, new_theta in enumerate(new_thetas):
            self.logger.record(f"train/theta_{idx}", new_theta.mean().item())
        new_theta = torch.max(new_thetas, dim=-1)[0]

        # Can't use env_steps b/c we are inside the learn function which is called only
        # every train_freq steps:
        if self._n_updates % self.theta_update_interval == 0:
            self.theta = self.tau_theta*self.theta + \
                (1 - self.tau_theta) * new_theta

    def learn(self, total_timesteps):
        # Start a timer to log fps:
        self.t0 = time.thread_time_ns()

        while self.env_steps < total_timesteps:
            state, _ = self.env.reset()
            if self.env_steps == 0:
                self.ref_state = state
            episode_reward = 0
            done = False
            # Random choice:
            action = self.online_logus.choose_action(state)

            self.num_episodes += 1
            self.rollout_reward = 0
            while not done:
                # take a random action:
                if self.env_steps < self.learning_starts:
                    action = self.env.action_space.sample()
                else:
                    action = self.online_logus.choose_action(state)

                next_state, reward, terminated, truncated, infos = self.env.step(
                    action)
                done = terminated or truncated
                self.rollout_reward += reward
                if self.env_steps == 0:
                    self.ref_action = action
                    self.ref_reward = reward
                    self.ref_next_state = next_state

                # TODO: Shorten this: (?)
                if (self.train_freq == -1 and terminated) or (self.train_freq != -1 and self.env_steps % self.train_freq == 0):
                    if self.env_steps > self.learning_starts:
                        self.train()

                if self.env_steps % self.target_update_interval == 0:
                    # Do a Polyak update of parameters:
                    self.target_logus.polyak(self.online_logus, self.tau)
                    # self.beta = 1/ ( 1 / self.beta - 1e-3)

                self.env_steps += 1

                episode_reward += reward
                self.replay_buffer.add(
                    state, next_state, action, reward, terminated, infos)
                state = next_state

                self._log_stats()

    def _log_stats(self):
        if self.env_steps == 0:
            # Log the hparams:
            self.logger.record("hparams/beta", self.beta)
            self.logger.record("hparams/learning_rate", self.learning_rate)
            self.logger.record("hparams/batch_size", self.batch_size)
            self.logger.record("hparams/buffer_size", self.buffer_size)
            self.logger.record("hparams/tau", self.tau)
            self.logger.record("hparams/tau_theta", self.tau_theta)
            self.logger.record("hparams/gradient_steps", self.gradient_steps)
            self.logger.record("hparams/hidden_dim", self.hidden_dim)
            self.logger.record("hparams/train_freq", self.train_freq)
            self.logger.record("hparams/max_grad_norm", self.max_grad_norm)
            self.logger.record("hparams/num_nets", self.num_nets)
            self.logger.record("hparams/target_update_interval",
                               self.target_update_interval)
            self.logger.record("hparams/theta_update_interval",
                               self.theta_update_interval)
            self.logger.record("hparams/actor_learning_rate",
                               self.actor_learning_rate)

        elif self.env_steps % self.log_interval == 0:
            # end timer:
            t_final = time.thread_time_ns()
            # fps averaged over log_interval steps:
            fps = self.log_interval / ((t_final - self.t0) / 1e9)

            avg_eval_rwd = self.evaluate()
            self.eval_auc += avg_eval_rwd
            if self.save_checkpoints:
                torch.save(self.online_logu.state_dict(),
                           'sql-policy.para')
            self.logger.record("time/env. steps", self.env_steps)
            self.logger.record("eval/avg_reward", avg_eval_rwd)
            self.logger.record("eval/auc", self.eval_auc)
            self.logger.record("time/num. episodes", self.num_episodes)
            self.logger.record("time/fps", fps)
            self.logger.record("rollout/beta", self.beta)
            if self.use_wandb:
                wandb.log({'env_steps': self.env_steps,'eval/avg_reward': avg_eval_rwd})
            self.logger.dump(step=self.env_steps)
            self.t0 = time.thread_time_ns()

        self.logger.record("rollout/reward", self.rollout_reward)

    def evaluate(self, n_episodes=1):
        # run the current policy and return the average reward
        avg_reward = 0.
        for ep in range(n_episodes):
            state, _ = self.eval_env.reset()
            done = False
            while not done:
                action = self.online_logus.greedy_action(state)
                # action = self.online_logus.choose_action(state)

                next_state, reward, terminated, truncated, info = self.eval_env.step(
                    action)
                avg_reward += reward
                state = next_state
                done = terminated or truncated
            # self.eval_env.close()

        avg_reward /= n_episodes
        return avg_reward


def main():
    env_id = 'CartPole-v1'
    # env_id = 'Taxi-v3'
    # env_id = 'CliffWalking-v0'
    # env_id = 'Acrobot-v1'
    # env_id = 'LunarLander-v2'
    # env_id = 'Pong-v'
    # env_id = 'FrozenLake-v1'
    # env_id = 'MountainCar-v0'
    # env_id = 'Drug-v0'

    from hparams import cartpole_hparams0 as config
    agent = LogULearner(env_id, **config, device='cpu',
                        log_dir='multinasium', num_nets=2, render=0)
    # agent = CustomDQN(env_id, device='cuda', **config)

    agent.learn(total_timesteps=200_000)


if __name__ == '__main__':
    for _ in range(1):
        main()
