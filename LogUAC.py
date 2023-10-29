from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape, get_flattened_obs_dim
import gymnasium as gym
import numpy as np
import torch
from torch.nn import functional as F
from torch import nn
import time

import wandb
from darer.Models import OnlineNets, Optimizers, TargetNets, LogUsa
from darer.utils import logger_at_folder
from stable_baselines3.sac.policies import Actor
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.torch_layers import MlpExtractor, FlattenExtractor
# raise warning level for debugger:
# import warnings
# warnings.filterwarnings("error")


class LogUActor:
    def __init__(self,
                 env_id,
                 beta,
                 learning_rate,
                 batch_size,
                 buffer_size,
                 target_update_interval,
                 theta_update_interval,
                 tau,
                 actor_learning_rate=None,
                 hidden_dim=64,
                 num_nets=2,
                 tau_theta=0.001,
                 learning_starts=5000,
                 gradient_steps=1,
                 train_freq=-1,
                 max_grad_norm=10,
                 device='cpu',
                 log_dir=None,
                 render=False,
                 log_interval=1000,
                 save_checkpoints=False,
                 ) -> None:
        self.env_id = env_id
        self.env = gym.make(env_id)
        self.vec_env = gym.make(env_id)
        # self.vec_env = gym.make_vec(self.env_id, num_envs=batch_size, vectorization_mode='sync')

        # make another instance for evaluation purposes only:
        self.eval_env = gym.make(env_id,
                                 render_mode='human' if render else None)

        self.nA = get_action_dim(self.env.action_space)
        self.nS = get_flattened_obs_dim(self.env.observation_space)
        self.beta = beta
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.theta_update_interval = theta_update_interval
        self.tau = tau
        self.hidden_dim = hidden_dim
        self.gradient_steps = gradient_steps
        self.learning_starts = learning_starts
        self.device = device
        self.save_checkpoints = save_checkpoints
        self.log_interval = log_interval
        self.tau_theta = tau_theta
        self.train_freq = train_freq
        self.max_grad_norm = max_grad_norm
        self.num_nets = num_nets
        self.prior = None
        self.actor_learning_rate = actor_learning_rate if \
            actor_learning_rate is not None else learning_rate
        
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
        self.logger = logger_at_folder(log_dir, algo_name=f'{env_id}')

        self._n_updates = 0
        self.env_steps = 0
        self._initialize_networks()

    def _initialize_networks(self):
        self.online_logus = OnlineNets([LogUsa(self.env,
                                               hidden_dim=self.hidden_dim,
                                               device=self.device)
                                        for _ in range(self.num_nets)])
        self.target_logus = TargetNets([LogUsa(self.env,
                                               hidden_dim=self.hidden_dim,
                                               device=self.device)
                                        for _ in range(self.num_nets)])
        self.target_logus.load_state_dict(
            [logu.state_dict() for logu in self.online_logus])
        self.actor = Actor(self.env.observation_space, self.env.action_space,
                           [self.hidden_dim, self.hidden_dim],
                           FlattenExtractor(self.env.observation_space),
                           self.nS,
                          )
        # send the actor to device:
        self.actor.to(self.device)
        # TODO: Try a fixed covariance network (no/ignored output)
        # Make (all) LogUs and Actor learnable:
        opts = [torch.optim.Adam(logu.parameters(), lr=self.learning_rate)
                for logu in self.online_logus]
        opts.append(torch.optim.Adam(
            self.actor.parameters(), lr=self.actor_learning_rate))
        self.optimizers = Optimizers(opts)

    def train(self,):
        # self.actor.set_training_mode(True)
        # average self.theta over multiple gradient steps
        new_thetas = torch.zeros(
            self.gradient_steps, self.num_nets).to(self.device)
        for grad_step in range(self.gradient_steps):
            replay = self.replay_buffer.sample(self.batch_size)
            states, actions, next_states, dones, rewards = replay
            actor_actions, curr_log_prob = self.actor.action_log_prob(states)
            with torch.no_grad():
                # use same number of samples as the batch size for convenience:
                prior_actions = np.array([self.vec_env.action_space.sample() for _ in range(self.batch_size)])
                # repeat the ref_next_state n_samples times:
                ref_next_state = torch.stack([torch.Tensor(self.ref_next_state) for _ in range(self.batch_size)], dim=1).T
                # Calculate ref_logu for all prior_actions at once
                ref_logu = [logu(ref_next_state, prior_actions) for logu in self.online_logus]
                ref_logu = torch.stack(ref_logu, dim=-1)
                ref_logu = ref_logu.mean(dim=0)
                ref_chi = torch.exp(ref_logu)
                ref_chi = ref_chi.squeeze()
                new_theta = self.ref_reward - torch.log(ref_chi)
                # Swap inf values with 0
                # new_theta[new_theta == float('inf')] = 0
                # new_theta[new_theta == float('-inf')] = 0
                new_thetas[grad_step, :] = new_theta

                # Action by the current actor for the sampled state
                # next_actions_pi, next_log_prob = self.actor.action_log_prob(
                #     next_states)
                # next_log_prob = next_log_prob.reshape(-1, 1)
                
                # rand_actions = self.vec_env.action_space.sample()
                
                target_next_logu = torch.stack([target_logu(next_states, prior_actions)
                                                for target_logu in self.target_logus], dim=-1)

                next_logu, _ = torch.min(target_next_logu, dim=-1)

                expected_curr_logu = self.beta * \
                    (rewards + self.theta) + (1 - dones) * next_logu
                expected_curr_logu = expected_curr_logu.squeeze(1)

            # next_logu = torch.cat([target_logu(next_states, next_actions_pi)
            #                        for target_logu in self.target_logus], dim=1)
            curr_logu = torch.stack([online_logu(states, actions)
                                   for online_logu in self.online_logus], dim=-1).squeeze(1)
            # print(curr_logu)

            self.logger.record("train/theta", self.theta.item())
            self.logger.record("train/avg logu", curr_logu.mean().item())
            # Huber loss:
            loss = 0.5*sum(F.smooth_l1_loss(logu, expected_curr_logu)
                           for logu in curr_logu.T)
            # MSE loss:
            actor_curr_logu = torch.stack([online_logu(states, actor_actions)
                                         for online_logu in self.online_logus], dim=-1)

            actor_loss = 0.5 * \
                F.smooth_l1_loss(curr_log_prob, actor_curr_logu.max(dim=-1)[0].squeeze())
            # actor_loss = -torch.mean(curr_log_prob - actor_curr_logu.min(dim=-1)[0].squeeze())
                
            self.logger.record("train/log_prob", curr_log_prob.mean().item())
            self.logger.record("train/loss", loss.item())
            self.logger.record("train/actor_loss", actor_loss.item())
            self.optimizers.zero_grad()
            # Increase update counter
            self._n_updates += self.gradient_steps

            # if self._n_updates % 100 == 0:
            actor_loss.backward()  

            # Clip gradient norm
            loss.backward()

            # Log the average gradient:
            for idx, logu in enumerate(self.online_logus.nets):
                norm = torch.max(torch.stack(
                [p.grad.detach().abs().max() for p in logu.parameters()]
                ))
                self.logger.record(f"grad/logu{idx}_norm", norm.item())
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)

            actor_norm = torch.max(torch.stack(
                [p.grad.detach().abs().max() for p in self.actor.parameters()]
            ))
            self.logger.record("grad/actor_norm", actor_norm.item())
            self.online_logus.clip_grad_norm(self.max_grad_norm)
            # Clip the actor's gradients
            self.optimizers.step()
            # record both thetas:
            for idx, theta in enumerate(new_theta.squeeze(0)):
                self.logger.record(f"train/theta_{idx}", theta.item())
        # TODO: Take the mean, then aggregate:
        new_theta = torch.max(new_thetas.mean(dim=0), dim=0)[0]

        if self._n_updates % self.theta_update_interval == 0:
            self.theta = self.tau_theta * self.theta + \
                (1 - self.tau_theta) * new_theta

    def learn(self, total_timesteps):
        # Start a timer to log fps:
        self.t0 = time.thread_time_ns()

        while self.env_steps < total_timesteps:
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            self.num_episodes += 1
            self.rollout_reward = 0
            while not done:
                self.actor.set_training_mode(False)
                if self.env_steps < self.learning_starts:
                    # take a random action:
                    action = self.env.action_space.sample()
                else:
                    action, _ = self.actor.predict(state)
                    # add some noise
                    # action += np.random.normal(0, 0.05, size=action.shape)#.clamp(-0.2, 0.2)
                    # assert that actions are in allowed range:
                    assert self.env.action_space.contains(action)

                next_state, reward, terminated, truncated, infos = self.env.step(
                    action)
                done = terminated or truncated
                self.rollout_reward += reward
                if self.env_steps == 0:
                    self.ref_state = state
                    self.ref_action = action
                    self.ref_reward = reward
                    self.ref_next_state = next_state

                if (self.train_freq == -1 and done) or (self.train_freq != -1 and self.env_steps % self.train_freq == 0):
                    if self.replay_buffer.size() > self.learning_starts:
                        self.train()

                if self.env_steps % self.target_update_interval == 0:
                    # Do a Polyak update of parameters:
                    self.target_logus.polyak(self.online_logus, self.tau)

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
            self.logger.record("hparams/target_update_interval", self.target_update_interval)
            self.logger.record("hparams/theta_update_interval", self.theta_update_interval)
            self.logger.record("hparams/actor_learning_rate", self.actor_learning_rate)

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
            self.logger.record("time/n_updates", self._n_updates)
            self.logger.record("time/fps", fps)
            # # Log network params:
            # for idx, logu in enumerate(self.online_logus.nets):
            #     for name, param in logu.named_parameters():
            #         self.logger.record(f"params/logu_{idx}/{name}",
            #                            param.data.mean().item())
            # for name, param in self.actor.named_parameters():
            #     self.logger.record(f"params/actor/{name}",
            #                        param.data.mean().item())

            self.logger.dump(step=self.env_steps)
            self.t0 = time.thread_time_ns()
            # Tell wandb to log the same things:
            wandb.log({'time/env. steps': self.env_steps})
            wandb.log({'eval/avg_reward': avg_eval_rwd})

        self.logger.record("rollout/reward", self.rollout_reward)

    def evaluate(self, n_episodes=3):
        # run the current policy and return the average reward
        avg_reward = 0.
        for ep in range(n_episodes):
            state, _ = self.eval_env.reset()
            done = False
            while not done:
                self.actor.set_training_mode(False)

                action, _ = self.actor.predict(observation=state)  # , deterministic=True)
                next_state, reward, terminated, truncated, info = self.eval_env.step(
                    action)
                avg_reward += reward
                state = next_state
                done = terminated or truncated
            # self.eval_env.close()

        avg_reward /= n_episodes
        return avg_reward

    def save_video(self):
        video_env = self.env_id
        gym.wrappers.monitoring.video_recorder.VideoRecorder(video_env, path='video.mp4')
        raise NotImplementedError

def main():
    # env_id = 'LunarLander-v2'
    env_id = 'Pendulum-v1'
    env_id = 'Hopper-v4'
    env_id = 'HalfCheetah-v4'
    # env_id = 'Ant-v4'
    # env_id = 'Simple-v0'
    from darer.hparams import cheetah_hparams2 as config
    agent = LogUActor(env_id, **config, device='cuda',
                      num_nets=2, learning_starts=5000, theta_update_interval=500,
                      actor_learning_rate=1e-4, log_dir='pend',
                      render=0, max_grad_norm=10, log_interval=500)
    agent.learn(total_timesteps=5_000_000)


if __name__ == '__main__':
    for _ in range(3):
        main()
