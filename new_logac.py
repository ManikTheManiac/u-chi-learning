from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape, get_flattened_obs_dim, preprocess_obs
import gymnasium as gym
import numpy as np
import torch
from torch.nn import functional as F
from torch import nn
import time

import wandb
from darer.Models import OnlineNets, Optimizers, TargetNets, LogUsa, GaussianPolicy
from darer.utils import logger_at_folder
from stable_baselines3.common.buffers import ReplayBuffer
# raise warning level for debugger:
import warnings
warnings.filterwarnings("error")
HPARAM_ATTRS = ['beta', 'learning_rate', 'batch_size', 'buffer_size',
                'target_update_interval', 'theta_update_interval', 'tau',
                'actor_learning_rate', 'hidden_dim', 'num_nets', 'tau_theta',
                'learning_starts', 'gradient_steps', 'train_freq', 'max_grad_norm']

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
                 train_freq=1,
                 max_grad_norm=10,
                 device='cpu',
                 log_dir=None,
                 render=False,
                 log_interval=1000,
                 save_checkpoints=False,
                 use_wandb=False,
                 ) -> None:
        self.env_id = env_id
        self.env = gym.make(env_id)
        self.n_samples = 1
        # self.vec_env = gym.make_vec(self.env_id, num_envs=self.n_samples)

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
        self.use_wandb = use_wandb
        self.actor_learning_rate = actor_learning_rate if \
            actor_learning_rate is not None else learning_rate
        
        self.replay_buffer = ReplayBuffer(buffer_size=buffer_size,
                                          observation_space=self.env.observation_space,
                                          action_space=self.env.action_space,
                                          n_envs=1,
                                          handle_timeout_termination=True,
                                          device=device)
        self.ref_action = None
        self.ref_state = None
        self.ref_reward = None
        self.theta = torch.Tensor([0]).to(self.device)
        self.eval_auc = 0
        self.num_episodes = 0

        # Set up the logger:
        self.logger = logger_at_folder(log_dir, algo_name=f'{env_id}')
        # Log the hparams:
        for key in HPARAM_ATTRS:
            self.logger.record(f"hparams/{key}", self.__dict__[key])
        self.logger.dump()

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
        self.target_logus.load_state_dicts(
            [logu.state_dict() for logu in self.online_logus])
        self.actor = GaussianPolicy(self.hidden_dim, 
                                    self.env.observation_space, self.env.action_space,
                                    use_action_bounds=False,
                                    device=self.device)
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
            actor_actions, curr_log_prob, means = self.actor.sample(states)
            curr_logu = torch.stack([online_logu(states, actions)
                                   for online_logu in self.online_logus], dim=-1).squeeze(1)
            with torch.no_grad():
                # use same number of samples as the batch size for convenience:
                sampled_action = self.env.action_space.sample()
                prior_actions = np.array([sampled_action for _ in range(self.batch_size)])
                prior_actions = torch.Tensor(prior_actions).to(self.device)
                # repeat the ref_next_state n_samples times:
                ref_next_state = torch.stack([torch.Tensor(self.ref_next_state) for _ in range(self.batch_size)], dim=1).T
                # Calculate ref_logu for all prior_actions at once
                # ref_logu = [logu(ref_next_state, prior_actions) for logu in self.online_logus]
                # ref_logu = torch.stack(ref_logu, dim=-1).mean(dim=0)
                ref_next_state = torch.stack([torch.Tensor(self.ref_next_state) for _ in range(self.batch_size)], dim=1).T
                ref_logu = torch.stack([logu(ref_next_state, prior_actions) for logu in self.online_logus], dim=-1)


                # Calculate ref_chi for all samples at once
                ref_chi = torch.exp(ref_logu).mean(dim=0)

                new_theta = self.ref_reward - torch.log(ref_chi)
                new_thetas[grad_step, :] = new_theta

                
                rand_actions = np.array([sampled_action for _ in range(self.batch_size)])
                rand_actions = torch.Tensor(rand_actions).to(self.device)               
                
                target_next_logu = torch.stack([target_logu(next_states, rand_actions)
                                                for target_logu in self.target_logus], dim=-1)

                next_logu, _ = torch.max(target_next_logu, dim=-1)
                next_logu = next_logu * (1 - dones) + self.theta * dones

                expected_curr_logu = self.beta * \
                    (rewards + self.theta) + (1 - dones) * next_logu
                
                
                # new_theta = torch.mean((self.beta * rewards + (1-dones)*next_logu - curr_logu) / -self.beta)
                # new_theta = torch.min(new_theta)
                expected_curr_logu = expected_curr_logu.squeeze(1)

            

            self.logger.record("train/theta", self.theta.item())
            self.logger.record("train/avg logu", curr_logu.mean().item())
            # Huber loss:
            loss = 0.5*sum(F.smooth_l1_loss(logu, expected_curr_logu)
                           for logu in curr_logu.T)
            # MSE loss:
            actor_curr_logu = torch.stack([online_logu(states, actor_actions)
                                         for online_logu in self.online_logus], dim=-1)

            # actor_loss = 0.5 * \
                # F.smooth_l1_loss(curr_log_prob, actor_curr_logu.max(dim=-1)[0])
            # PPO clips the prioritzed sampling
            # ratio = torch.exp(curr_logu - actor_curr_logu.min(dim=-1)[0] )
            # Clip the ratio:
            # eps=0.2
            # ratio = torch.clamp(ratio, 1-eps, 1+eps)
            # actor_loss = torch.log(ratio)
                
            actor_loss = -(curr_log_prob - actor_curr_logu.min(dim=-1)[0]).mean()

            self.logger.record("train/log_prob", curr_log_prob.mean().item())
            self.logger.record("train/loss", loss.item())
            self.logger.record("train/actor_loss", actor_loss.item())
            self.optimizers.zero_grad()
            # Increase update counter
            self._n_updates += self.gradient_steps

            # if self._n_updates % 100 == 0:
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            
            # Clip gradient norm
            loss.backward()
            self.online_logus.clip_grad_norm(self.max_grad_norm)

            # Log the average gradient:
            for idx, logu in enumerate(self.online_logus.nets):
                norm = torch.max(torch.stack(
                [p.grad.detach().abs().max() for p in logu.parameters()]
                ))
                self.logger.record(f"grad/logu{idx}_norm", norm.item())
            actor_norm = torch.max(torch.stack(
                [p.grad.detach().abs().max() for p in self.actor.parameters()]
            ))
            self.logger.record("grad/actor_norm", actor_norm.item())
            self.optimizers.step()
            # record both thetas:
            for idx, theta in enumerate(new_theta.squeeze(0)):
                self.logger.record(f"train/theta_{idx}", theta.item())
        # TODO: Take the mean, then aggregate:
        # new_theta = new_theta 
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
                # self.actor.set_training_mode(False)
                if self.env_steps < self.learning_starts:
                    # take a random action:
                    noisy_action = self.env.action_space.sample()
                else:
                    with torch.no_grad():
                        noisy_action, logprob, _ = self.actor.sample(state)
                        # log the logprob:
                        # self.logger.record("rollout/log_prob", logprob.mean().item())
                        noisy_action = noisy_action.cpu().numpy()
                next_state, reward, terminated, truncated, infos = self.env.step(
                    noisy_action)
                done = terminated or truncated
                self.rollout_reward += reward
                if self.env_steps == 0:
                    self.ref_state = state
                    self.ref_action = noisy_action
                    self.ref_reward = reward
                    self.ref_next_state = next_state

                if (self.train_freq == -1 and done) or (self.train_freq != -1 and self.env_steps % self.train_freq == 0):
                    if self.replay_buffer.size() > self.batch_size:
                        self.train()

                if self.env_steps % self.target_update_interval == 0:
                    # Do a Polyak update of parameters:
                    self.target_logus.polyak(self.online_logus, self.tau)
                    
                # if self.env_steps % 10_000 == 0:
                #     self.beta = min(1e3, self.beta * 1.4)

                self.env_steps += 1

                episode_reward += reward
                infos = [infos]
                self.replay_buffer.add(
                    state, next_state, noisy_action, reward, terminated, infos)
                state = next_state
                
                self._log_stats()
            if done:
                self.logger.record("rollout/reward", self.rollout_reward)


    def _log_stats(self):
        if self.env_steps % self.log_interval == 0:
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
            self.logger.record("rollout/beta", self.beta)
            self.logger.record("time/num. episodes", self.num_episodes)
            self.logger.record("time/n_updates", self._n_updates)
            self.logger.record("time/fps", fps)
            # Log network params:
            # for idx, logu in enumerate(self.online_logus.nets):
            #     for name, param in logu.named_parameters():
            #         self.logger.record(f"params/logu_{idx}/{name}",
            #                            param.data.mean().item())
            # for name, param in self.actor.named_parameters():
            #     self.logger.record(f"params/actor/{name}",
            #                        param.data.mean().item())

            self.logger.dump(step=self.env_steps)
            if self.use_wandb:
                wandb.log({'env step': self.env_steps, 'avg_eval_rwd': avg_eval_rwd})
            self.t0 = time.thread_time_ns()


    def evaluate(self, n_episodes=1):
        # run the current policy and return the average reward
        avg_reward = 0.
        for ep in range(n_episodes):
            state, _ = self.eval_env.reset()
            done = False
            while not done:
                # self.actor.set_training_mode(False)
                with torch.no_grad():
                    noisyaction, logprob, action = self.actor.sample(state)  # , deterministic=True)
                    action = action.cpu().numpy()
                next_state, reward, terminated, truncated, info = self.eval_env.step(
                    action)
                avg_reward += reward
                state = next_state
                done = terminated or truncated

        avg_reward /= n_episodes
        return avg_reward

    def save_video(self):
        video_env = self.env_id
        gym.wrappers.monitoring.video_recorder.VideoRecorder(video_env, path='video.mp4')
        raise NotImplementedError

def main():
    # env_id = 'LunarLanderContinuous-v2'
    # env_id = 'BipedalWalker-v3'
    # env_id = 'CartPole-v1'
    env_id = 'Pendulum-v1'
    # env_id = 'Hopper-v4'
    env_id = 'HalfCheetah-v4'
    # env_id = 'Ant-v4'
    # env_id = 'Simple-v0'
    from darer.hparams import cheetah_hparams2 as config
    agent = LogUActor(env_id, **config, device='cuda',
                      num_nets=2, log_dir='pend', theta_update_interval=200,
                      render=0, max_grad_norm=10, log_interval=1000)
    agent.learn(total_timesteps=5_000_000)


if __name__ == '__main__':
    for _ in range(3):
        main()
