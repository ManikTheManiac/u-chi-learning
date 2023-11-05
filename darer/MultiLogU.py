import gymnasium as gym
import torch
from torch.nn import functional as F
import time
from stable_baselines3.common.buffers import ReplayBuffer
import wandb
from Models import LogUNet, OnlineNets, Optimizers, TargetNets
from utils import env_id_to_envs, logger_at_folder
HPARAM_ATTRS = {'beta', 'learning_rate', 'batch_size', 'buffer_size', 
                'target_update_interval', 'tau', 'theta_update_interval', 
                'hidden_dim', 'num_nets', 'tau_theta', 'gradient_steps',
                'train_freq', 'max_grad_norm', 'learning_starts'}

str_to_aggregator = {'min': torch.min, 'max': torch.max, 'mean': torch.mean}
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
                 loss_fn=None,
                 device='cpu',
                 render=False,
                 log_dir=None,
                 log_interval=1000,
                 save_checkpoints=False,
                 use_wandb=False,
                 aggregator='max'
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
        self.aggregator = aggregator
        self.aggregator_fn = str_to_aggregator[aggregator]

        self.replay_buffer = ReplayBuffer(buffer_size=buffer_size,
                                          observation_space=self.env.observation_space,
                                          action_space=self.env.action_space,
                                          n_envs=1,
                                          handle_timeout_termination=False,
                                          device=device)
        self.nA = self.env.action_space.n
        self.ref_action = None
        self.ref_state = None
        self.ref_reward = None
        self.theta = torch.Tensor([0]).to(self.device)
        self.eval_auc = 0
        self.num_episodes = 0

        # Set up the logger:
        self.logger = logger_at_folder(log_dir, algo_name=f'{aggregator}-theta')
        # Log the hparams:
        for key in HPARAM_ATTRS:
            self.logger.record(f"hparams/{key}", self.__dict__[key])
        self.logger.dump()

        self._n_updates = 0
        self.env_steps = 0
        self._initialize_networks()
        self.loss_fn = F.smooth_l1_loss if loss_fn is None else loss_fn


    def _initialize_networks(self):
        self.online_logus = OnlineNets([LogUNet(self.env, hidden_dim=self.hidden_dim, device=self.device)
                                                     for _ in range(self.num_nets)],
                                                     aggregator=self.aggregator)
        
        self.target_logus = TargetNets([LogUNet(self.env, hidden_dim=self.hidden_dim, device=self.device)
                                                     for _ in range(self.num_nets)])
        self.target_logus.load_state_dicts([logu.state_dict() for logu in self.online_logus])
        # Make (all) LogUs learnable:
        opts = [torch.optim.Adam(logu.parameters(), lr=self.learning_rate)
                for logu in self.online_logus]
        self.optimizers = Optimizers(opts)


    def train(self):
        # average self.theta over multiple gradient steps
        new_thetas = torch.zeros(self.gradient_steps, self.num_nets).to(self.device)
        for grad_step in range(self.gradient_steps):
            # Sample a batch from the replay buffer:
            batch = self.replay_buffer.sample(self.batch_size)
            states, actions, next_states, dones, rewards = batch
            # Calculate the current logu values (feedforward):
            curr_logu = torch.cat([online_logu(states).squeeze().gather(1, actions.long())
                                   for online_logu in self.online_logus], dim=1)
            
            with torch.no_grad():
                ref_logus = [logu(self.ref_next_state)
                            for logu in self.online_logus]
                # since pi0 is same for all, just do exp(ref_logu) and sum over actions:
                ref_chi = torch.stack([torch.exp(ref_logu_val).sum(dim=-1) / self.nA
                                       for ref_logu_val in ref_logus], dim=-1)
                new_thetas[grad_step, :] = self.ref_reward - torch.log(ref_chi)

                #TODO: this looks wrong (expectation of logu?? should be of u)
                target_next_logus = [target_logu(next_states).clamp(-30, 30)
                                        for target_logu in self.target_logus]
                # log target logu min and max:

                self.logger.record("train/target_min_logu", target_next_logus[0].min().item())
                self.logger.record("train/target_max_logu", target_next_logus[0].max().item())
                target_next_u = torch.stack([torch.exp(target_logu).sum(dim=-1) / self.nA
                                            for target_logu in target_next_logus], dim=-1)
                target_next_logu = torch.log(target_next_u)

                next_logu, _ = self.aggregator_fn(target_next_logu, dim=1)
                if isinstance(self.env.observation_space, gym.spaces.Discrete):
                    pass
                else:
                    next_logu = next_logu.unsqueeze(1)
                # When an episode terminates, next_logu should be theta or zero?:
                assert next_logu.shape == dones.shape
                next_logu = next_logu * (1 - dones) #+ self.theta * dones

                # "Backup" eigenvector equation:
                expected_curr_logu = self.beta * (rewards + self.theta) + next_logu
                expected_curr_logu = expected_curr_logu.squeeze(1)
                # clamp the logu values to avoid overflow:
                # expected_curr_logu = torch.clamp(expected_curr_logu, -10, 10)
                # calculate theta in a similar way:
                # next_chi = torch.stack([torch.exp(next_logu_val).sum(dim=-1) / self.env.action_space.n
                # new_thetas = rewards + 1/self.beta * (next_chi - curr_logu)


            expected_curr_logu = torch.clamp(expected_curr_logu, -30, 30)
            curr_logu = torch.clamp(curr_logu, -30, 30)
            self.logger.record("train/theta", self.theta.item())
            self.logger.record("train/avg logu", curr_logu.mean().item())
            self.logger.record("train/min logu", curr_logu.min().item())
            self.logger.record("train/max logu", curr_logu.max().item())

            # Calculate the logu ("critic") loss:
            loss = 0.5*sum(self.loss_fn(logu, expected_curr_logu)
                           for logu in curr_logu.T)

            self.logger.record("train/loss", loss.item())
            self.optimizers.zero_grad()
            # Increase update counter
            self._n_updates += self.gradient_steps

            # Clip gradient norm
            loss.backward()
            self.online_logus.clip_grad_norm(self.max_grad_norm)

            # Log the max gradient:
            total_norm = torch.max(torch.stack(
                        [px.grad.detach().abs().max() 
                         for p in self.online_logus.parameters() for px in p]
                        ))
            self.logger.record("train/max_grad", total_norm.item())
            self.optimizers.step()
        # new_thetas = torch.clamp(new_thetas, 1, -1)
        # Log both theta values:
        for idx, new_theta in enumerate(new_thetas.T):
            self.logger.record(f"train/theta_{idx}", new_theta.mean().item())
        new_theta = self.aggregator_fn(new_thetas.mean(dim=0), dim=0)[0]

        # Can't use env_steps b/c we are inside the learn function which is called only
        # every train_freq steps:
        if self._n_updates % self.theta_update_interval == 0:
            self.theta = self.tau_theta * self.theta + (1 - self.tau_theta) * new_theta

    def learn(self, total_timesteps):
        # Start a timer to log fps:
        self.t0 = time.thread_time_ns()

        while self.env_steps < total_timesteps:
            state, _ = self.env.reset()
            if self.env_steps == 0:
                self.ref_state = state
            episode_reward = 0
            done = False
            self.num_episodes += 1
            self.rollout_reward = 0
            while not done:
                # take a random action:
                if self.env_steps < self.learning_starts:
                    action = self.env.action_space.sample()
                else:
                    action = self.online_logus.choose_action(state)
                    # action = self.online_logus.greedy_action(state)
                    # action = self.env.action_space.sample()

                next_state, reward, terminated, truncated, infos = self.env.step(
                    action)
                done = terminated or truncated
                self.rollout_reward += reward
                if self.env_steps == 0:
                    self.ref_action = action
                    self.ref_reward = reward
                    self.ref_next_state = next_state

                train_this_step = (self.train_freq == -1 and terminated) or \
                    (self.train_freq != -1 and self.env_steps % self.train_freq == 0)
                if train_this_step:
                    if self.env_steps > self.batch_size:#self.learning_starts:
                        self.train()

                if self.env_steps % self.target_update_interval == 0:
                    # Do a Polyak update of parameters:
                    self.target_logus.polyak(self.online_logus, self.tau)

                    # Update beta:
                    # self.beta = min(20, self.beta+1e-2)

                #TODO: implement beta schedule
                self.env_steps += 1
                episode_reward += reward

                # Add the transition to the replay buffer:
                sarsa = (state, next_state, action, reward, terminated)
                self.replay_buffer.add(*sarsa, [infos])
                state = next_state
                self._log_stats()

            if done:
                self.logger.record("rollout/reward", self.rollout_reward)


    def _log_stats(self):
        if self.env_steps % self.log_interval == 0:
            # end timer:
            t_final = time.thread_time_ns()
            # fps averaged over log_interval steps:
            fps = self.log_interval / ((t_final - self.t0 + 1e-8) / 1e9)

            if self.env_steps >= 0:
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
            # self.logger.record("train/lr", self.optimizers.)
            if self.use_wandb:
                wandb.log({'env_steps': self.env_steps,'eval/avg_reward': avg_eval_rwd})
            self.logger.dump(step=self.env_steps)
            self.t0 = time.thread_time_ns()


    def evaluate(self, n_episodes=2):
        # run the current policy and return the average reward
        avg_reward = 0.
        # log the action frequencies:
        action_freqs = torch.zeros(self.nA)
        for ep in range(n_episodes):
            state, _ = self.eval_env.reset()
            done = False
            n_steps = 0
            while not done:
                action = self.online_logus.greedy_action(state)
                # action = self.online_logus.choose_action(state)
                action_freqs[action] += 1
                action = action.item()
                # action = self.online_logus.choose_action(state)
                n_steps += 1

                next_state, reward, terminated, truncated, info = self.eval_env.step(
                    action)
                avg_reward += reward
                state = next_state
                done = terminated or truncated

        avg_reward /= n_episodes
        # log the action frequencies:
        action_freqs /= n_episodes
        for i, freq in enumerate(action_freqs):
            self.logger.record(f'eval/action_freq_{i}', freq.item())
        return avg_reward


def main():
    env_id = 'CartPole-v1'
    # env_id = 'Taxi-v3'
    # env_id = 'CliffWalking-v0'
    env_id = 'Acrobot-v1'
    env_id = 'LunarLander-v2'
    env_id = 'Pong-v4'
    # env_id = 'FrozenLake-v1'
    # env_id = 'MountainCar-v0'
    # env_id = 'Drug-v0'
    from hparams import lunar_hparams as config
    agent = LogULearner(env_id, **config, device='cpu', log_interval=500,
                        log_dir='pend', num_nets=2, render=1, aggregator='max')

    agent.learn(total_timesteps=1_000_000)


if __name__ == '__main__':
    for _ in range(1):
        main()
