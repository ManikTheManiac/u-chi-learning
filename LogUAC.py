import gymnasium as gym
import numpy as np
import torch
from torch.nn import functional as F
from torch import nn
import time
from darer.ReplayBuffers import Memory, SB3Memory
from darer.Models import OnlineNets, Optimizers, TargetNets, LogUsa
from darer.utils import logger_at_folder
from stable_baselines3.sac.policies import SACPolicy, Actor
from stable_baselines3.common.buffers import ReplayBuffer
# raise warning level for debugger:
import warnings
warnings.filterwarnings("error")
from stable_baselines3.common.preprocessing import preprocess_obs, get_action_dim


class LogUActor:
    def __init__(self,
                 env_id,
                 beta,
                 learning_rate,
                 batch_size,
                 buffer_size,
                 target_update_interval,
                 tau,
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
        # make another instance for evaluation purposes only:
        self.eval_env = gym.make(env_id, render_mode='human' if render else None)

        # from stable_baselines3.common.vec_env.util import unwrap_vec_normalize
        # self._vec_normalize_env = unwrap_vec_normalize(self.env)
        self.nA = get_action_dim(self.env.action_space)
        self.beta = beta
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
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
        self.logger = logger_at_folder(
            log_dir, algo_name=f'hp2{num_nets}nets')

        self._n_updates = 0
        self.env_steps = 0
        self._initialize_networks()

    def _initialize_networks(self):
        self.online_logus = OnlineNets(list_of_nets=[LogUsa(self.env, hidden_dim=self.hidden_dim, device=self.device)
                                                     for _ in range(self.num_nets)])
        self.target_logus = TargetNets(list_of_nets=[LogUsa(self.env, hidden_dim=self.hidden_dim, device=self.device)
                                                     for _ in range(self.num_nets)])
        self.target_logus.load_state_dict(
            [logu.state_dict() for logu in self.online_logus])
        n_features = self.env.observation_space.shape[0]
        self.actor = Actor(self.env.observation_space, self.env.action_space, 
                           [self.hidden_dim, self.hidden_dim], 
                           features_extractor=nn.Flatten(),
                           features_dim=n_features,)
        # send the actor to device:
        self.actor.to(self.device)
        #TODO: Try a fixed covariance network (no/ignored output)
                        #    device=self.device)
        # Make (all) LogUs and Actor learnable:
        opts = [torch.optim.Adam(logu.parameters(), lr=self.learning_rate)
                for logu in self.online_logus]
        opts.append(torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate))
        self.optimizers = Optimizers(opts)

    def train(self,):
        # replay = self.replay_buffer.sample(self.batch_size, env=self._vec_normalize_env)
        # average self.theta over multiple gradient steps
        new_thetas = torch.zeros(self.gradient_steps, self.num_nets).to(self.device)
        for grad_step in range(self.gradient_steps):
            replay = self.replay_buffer.sample(self.batch_size)
            states, actions, next_states, dones, rewards = replay
            actor_actions, curr_log_prob = self.actor.action_log_prob(states)
            with torch.no_grad():
                # TODO: Average over many random actions:
                # n_samples = 10
                # prior_actions = np.array([[self.env.action_space.sample() for _ in range(self.num_nets)] 
                #                           for _ in range(n_samples)]).squeeze()
                # ref_next_state = torch.stack([torch.Tensor(self.ref_next_state) for _ in range(n_samples)], dim=1).squeeze()
                # prior_actions = torch.Tensor(prior_actions.T).to(self.device)
                # ref_logu = [logu(ref_next_state.unsqueeze(1), a_s.unsqueeze(1)) 
                #             for logu, a_s in zip(self.online_logus, prior_actions)]
                # ref_chi = torch.stack([torch.exp(ref_logu_val).sum(dim=0)
                #                        for ref_logu_val in ref_logu], dim=-1) /  n_samples
                n_samples = 2
                chi = torch.zeros(n_samples, self.num_nets).to(self.device)
                for i in range(n_samples):
                    prior_actions = np.array([self.env.action_space.sample() for _ in range(self.num_nets)])
                    ref_logu = [logu(self.ref_next_state, a)
                                for logu, a in zip(self.online_logus, prior_actions)]
                    ref_chi = torch.stack([torch.exp(ref_logu_val)
                                           for ref_logu_val in ref_logu], dim=-1)
                    chi[i] = ref_chi
                    
                ref_chi = chi.mean(dim=0)
                # prior_actions = np.array([self.env.action_space.sample() for _ in range(self.num_nets)])
                # ref_logu = [logu(self.ref_next_state, a) 
                #             for logu, a in zip(self.online_logus, prior_actions)]
                # ref_chi = torch.stack([torch.exp(ref_logu_val)
                #                        for ref_logu_val in ref_logu], dim=-1)
                new_theta = self.ref_reward - torch.log(ref_chi)

                new_thetas[grad_step, :] = new_theta

                # Action by the current actor for the sampled state
                next_actions_pi, next_log_prob = self.actor.action_log_prob(next_states)
                next_log_prob = next_log_prob.reshape(-1, 1)
                ###
                rand_actions = np.array([self.env.action_space.sample() for _ in range(next_states.shape[0])])
                ###
                target_next_logu = torch.stack([target_logu(next_states, rand_actions)
                                              for target_logu in self.target_logus], dim=1)

                next_logu, _ = torch.min(target_next_logu, dim=1)
                ###
                # for n in range(n_samples):
                #     rand_actions = np.array([self.env.action_space.sample() for _ in range(next_states.shape[0])])
                #     target_next_logu = torch.stack([target_logu(next_states, rand_actions)
                #                                     for target_logu in self.target_logus], dim=1)
                    
                # Need to use importance sampling to get the correct expectation:
                # next_logu *= torch.exp(-next_log_prob)# + np.log(self.nA))

                expected_curr_logu = self.beta * (rewards + self.theta) + (1 - dones) * next_logu
                expected_curr_logu = expected_curr_logu.squeeze(1)

            next_logu = torch.cat([target_logu(next_states, next_actions_pi)
                                      for target_logu in self.target_logus], dim=1)
            curr_logu = torch.cat([online_logu(states, actions)
                                   for online_logu in self.online_logus], dim=1)
            
            self.logger.record("train/theta", self.theta.item())
            self.logger.record("train/avg logu", curr_logu.mean().item())
            # Huber loss:
            loss = 0.5*sum(F.smooth_l1_loss(logu, expected_curr_logu) for logu in curr_logu.T)
            # MSE loss:
            actor_curr_logu = torch.cat([online_logu(states, actor_actions)
                                   for online_logu in self.online_logus], dim=1)

            actor_loss = 0.5*F.smooth_l1_loss(curr_log_prob , actor_curr_logu.min(dim=1)[0])
            self.logger.record("train/log_prob", curr_log_prob.mean().item())
            self.logger.record("train/loss", loss.item())
            self.logger.record("train/actor_loss", actor_loss.item())
            self.optimizers.zero_grad()
            # Increase update counter
            self._n_updates += self.gradient_steps

            actor_loss.backward()

            # Clip gradient norm
            loss.backward()

            # Log the average gradient:
            total_norm = torch.max(torch.stack(
                        [p.grad.detach().abs().max() for logu in self.online_logus.nets 
                         for p in logu.parameters()]
                        ))
            self.online_logus.clip_grad_norm(self.max_grad_norm)
            self.logger.record("max_grad", total_norm.item())
            self.optimizers.step()
            # record both thetas:
            for idx, theta in enumerate(new_theta.squeeze(0)):
                self.logger.record(f"train/theta_{idx}", theta.item())
        # new_thetas = torch.clamp(new_thetas, 0, -10)
        #TODO: Take the mean, then aggregate:
        new_theta = torch.min(new_thetas.mean(dim=0), dim=0)[0]

        # new_theta = torch.clamp(new_theta, min=0)
        # if self.env_steps % self.target_update_interval == 0:
        self.theta = self.tau_theta * self.theta + (1 - self.tau_theta) * new_theta

    def learn(self, total_timesteps):
        # Start a timer to log fps:
        t0 = time.thread_time_ns()
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
        
        while self.env_steps < total_timesteps:
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            # Random choice:
            action, _ = self.actor.predict(state, deterministic=True)
            # action = self.env.action_space.sample()
            self.num_episodes += 1
            self.rollout_reward = 0
            while not done:
                # take a random action:
                # action = self.env.action_space.sample()
                if self.env_steps < self.learning_starts:
                    action = self.env.action_space.sample()
                else:
                    action, _ = self.actor.predict(state)#, deterministic=True)

                next_state, reward, terminated, truncated, infos = self.env.step(action)
                done = terminated or truncated
                self.rollout_reward += reward
                if self.env_steps == 0:
                    self.ref_state = state
                    self.ref_action = action
                    self.ref_reward = reward
                    self.ref_next_state = next_state

                # TODO: Shorten this: (?)
                if (self.train_freq == -1 and done) or (self.train_freq != -1 and self.env_steps % self.train_freq == 0):
                    if self.replay_buffer.size() > self.learning_starts:
                        self.train()

                if self.env_steps % self.target_update_interval == 0:
                    # Do a Polyak update of parameters:
                    self.target_logus.polyak(self.online_logus, self.tau)

                self.env_steps += 1
                # next_action, _ = self.actor.predict(next_state,deterministic=False)

                episode_reward += reward
                #TODO: Determine whether this should be done or terminated (or truncated?)
                # Looks like done (both) works best... possibly because we need continuing env?
                # _ = self.env.action_space.sample()
                self.replay_buffer.add(
                    state, next_state, action, reward, terminated, infos)
                state = next_state
                # action = next_action
                if self.env_steps % self.log_interval == 0:
                    # end timer:
                    t_final = time.thread_time_ns()
                    # fps averaged over log_interval steps:
                    fps = self.log_interval / ((t_final - t0) / 1e9)

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

                    self.logger.dump(step=self.env_steps)
                    t0 = time.thread_time_ns()

                self.logger.record("rollout/reward", self.rollout_reward)


    def evaluate(self, n_episodes=3):
        # run the current policy and return the average reward
        avg_reward = 0.
        for ep in range(n_episodes):
            state, _ = self.eval_env.reset()
            done = False
            while not done:
                action, _ = self.actor.predict(state, deterministic=True)
                next_state, reward, terminated, truncated, info = self.eval_env.step(action)
                avg_reward += reward
                state = next_state
                done = terminated or truncated
            # self.eval_env.close()

        avg_reward /= n_episodes
        return avg_reward
    
    # def save_video(self):
        # video_env = self.env_id
        # gym.wrappers.monitoring.video_recorder.VideoRecorder(video_env, path='video.mp4')

def main():
    # env_id = 'LunarLander-v2'
    env_id = 'Pendulum-v1'
    # env_id = 'Hopper-v4'
    env_id = 'HalfCheetah-v4'
    # env_id = 'Ant-v4'
    # env_id = 'Simple-v0'
    from darer.hparams import easy_hparams2 as config
    agent = LogUActor(env_id, **config, device='cpu', log_dir='pend', 
                      num_nets=2, learning_starts=500, log_interval=500, render=0, max_grad_norm=10)
    agent.learn(total_timesteps=10_000_000)


if __name__ == '__main__':
    for _ in range(3):
        main()
