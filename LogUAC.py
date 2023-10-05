import gymnasium as gym
import torch
import numpy as np
from torch.nn import functional as F
from torch import nn
from gymnasium.wrappers import TimeLimit
import time
from darer.ReplayBuffers import Memory, SB3Memory
from darer.Models import OnlineNets, Optimizers, TargetNets
from darer.utils import logger_at_folder
from stable_baselines3.sac.policies import SACPolicy, Actor
# raise warning level for debugger:
import warnings
warnings.filterwarnings("error")
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.preprocessing import preprocess_obs


class LogUsa(nn.Module):
    def __init__(self, env, hidden_dim=256, device='cuda'):
        super(LogUsa, self).__init__()
        self.env = env
        self.device = device
        try:
            self.nS = env.observation_space.n
        except AttributeError:
            self.nS = env.observation_space.shape[0]

        self.nA = env.action_space.shape[0]
        self.fc1 = nn.Linear(self.nS + self.nA, hidden_dim, device=self.device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, device=self.device)
        self.fc3 = nn.Linear(hidden_dim, self.nA, device=self.device)
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
        logu = self.fc3(x)
        return logu


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
                 gradient_steps=1,
                 train_freq=-1,
                 max_grad_norm=10,
                 device='cpu',
                 log_dir=None,
                 log_interval=1000,
                 save_checkpoints=False,
                 ) -> None:
        self.env = gym.make(env_id)
        # timelimit:
        self.env = TimeLimit(self.env, max_episode_steps=500)
        # make another instance for evaluation purposes only:
        self.eval_env = gym.make(env_id)
        # self.eval_env = TimeLimit(self.eval_env, max_episode_steps=500)
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
        self.num_nets = num_nets
        self.prior = None

        # self.replay_buffer = Memory(buffer_size, device=device)
        self.replay_buffer = SB3Memory(buffer_size=buffer_size,
                                        observation_space=self.env.observation_space,
                                        action_space=self.env.action_space,
                                        n_envs=1,
                                        device=device)
        self.ref_action = None
        self.ref_state = None
        self.ref_reward = None
        self.theta = torch.Tensor([0]).to(self.device)
        self.eval_auc = 0
        self.num_episodes = 0
        # self.replay_buffer = ReplayBuffer(buffer_size)

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
        self.actor = Actor(self.env.observation_space, self.env.action_space, 
                           [self.hidden_dim, self.hidden_dim], 
                           features_extractor=nn.Flatten(),
                           features_dim=3,)
        # SACPolicy(self.env.observation_space, self.env.action_space, lambda x: 1e-3)#, self.hidden_dim, self.device)
        # Make (all) LogUs learnable:
        opts = [torch.optim.Adam(logu.parameters(), lr=self.learning_rate)
                for logu in self.online_logus]
        opts.append(torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate))
        self.optimizers = Optimizers(opts)

    def train(self,):
        # replay = self.replay_buffer.sample(self.batch_size, env=self._vec_normalize_env)
        # average self.theta over multiple gradient steps
        new_thetas = torch.zeros(self.gradient_steps)
        for grad_step in range(self.gradient_steps):
            replay = self.replay_buffer.sample(self.batch_size)
            states, actions, next_states, next_actions, dones, rewards = replay
            _, curr_log_prob = self.actor.action_log_prob(states)
            with torch.no_grad():
                ref_logu = [logu(self.ref_next_state, self.ref_action) for logu in self.online_logus]
                # since pi0 is same for all, just do exp(ref_logu) and sum over actions:
                ref_chi = torch.stack([torch.exp(ref_logu_val).sum(dim=-1)
                                       for ref_logu_val in ref_logu], dim=-1)
                new_theta = self.ref_reward - torch.log(ref_chi)
                new_thetas[grad_step] = torch.min(new_theta,dim=-1)[0]


                # Action by the current actor for the sampled state
                next_actions_pi, log_prob = self.actor.action_log_prob(next_states)
                log_prob = log_prob.reshape(-1, 1)
                ###
                target_next_logu = torch.cat([target_logu(next_states, next_actions_pi)
                                              for target_logu in self.target_logus], dim=1)

                next_logu, _ = torch.min(target_next_logu, dim=-1, keepdim=True)
          
                expected_curr_logu = self.beta * \
                    (rewards + self.theta) + (1 - dones) * next_logu
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
            # loss = F.mse_loss(curr_logu, expected_curr_logu)
            # actor_loss = (curr_log_prob/self.beta - curr_logu.min(dim=1)[0]).mean()
            actor_loss = F.mse_loss(curr_log_prob, expected_curr_logu)

            self.logger.record("train/loss", loss.item())
            self.logger.record("train/actor_loss", actor_loss.item())
            self.optimizers.zero_grad()
            # Increase update counter
            self._n_updates += self.gradient_steps

            actor_loss.backward()

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

        self.theta = self.tau_theta*self.theta + \
            (1 - self.tau_theta) * torch.mean(new_thetas)

    def learn(self, total_timesteps):
        # Start a timer to log fps:
        t0 = time.thread_time_ns()

        while self.env_steps < total_timesteps:
            state, _ = self.env.reset()
            if self.env_steps == 0:
                self.ref_state = state
            episode_reward = 0
            done = False
            # Random choice:

            action, _ = self.actor.predict(state)
            # Try 100 random actions, choose one with best logu:
            # rand_actions = np.array([self.env.action_space.sample()
            #                 for _ in range(100)])
            # rand_actions = torch.Tensor(rand_actions).to(self.device)
            # states = torch.Tensor(np.array([state for _ in range(100)])).to(self.device)
            # rand_logu = torch.cat([logu(states, rand_actions)
            #                           for logu in self.online_logus], dim=1)
            # rand_logu = rand_logu.min(dim=0)[0]
            # action = np.array([rand_actions[torch.argmax(rand_logu)].item())
        
            # print(action)
            self.num_episodes += 1
            self.rollout_reward = 0
            while not done:
                # take a random action:
                # action = self.env.action_space.sample()
                next_state, reward, terminated, truncated, infos = self.env.step(action)
                done = terminated or truncated
                self.rollout_reward += reward
                if self.env_steps == 0:
                    self.ref_action = action
                    self.ref_reward = reward
                    self.ref_next_state = next_state

                # TODO: Shorten this: (?)
                if (self.train_freq == -1 and terminated) or (self.train_freq != -1 and self.env_steps % self.train_freq == 0):
                    if self.replay_buffer.size() > self.batch_size:  # or learning_starts?
                        self.train()

                if self.env_steps % self.target_update_interval == 0:
                    # Do a Polyak update of parameters:
                    self.target_logus.polyak(self.online_logus, self.tau)
                    # loop thru the nets and do polyak manually:
                    # polyak_update(self.online_logus.nets[0].parameters(), self.target_logus.nets[0].parameters(), self.tau)
                    # polyak_update(self.online_logus.nets[1].parameters(), self.target_logus.nets[1].parameters(), self.tau)

                self.env_steps += 1
                next_action, _ = self.actor.predict(next_state)

                episode_reward += reward
                #TODO: Determine whether this should be done or terminated (or truncated?)
                # Looks like done (both) works best... possibly because we need continuing env?
                self.replay_buffer.add(
                    state, next_state, action, next_action, reward, done)
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

    def evaluate(self, n_episodes=5):
        # run the current policy and return the average reward
        avg_reward = 0.
        for ep in range(n_episodes):
            state, _ = self.eval_env.reset()
            done = False
            while not done:
                # action = self.online_logus.greedy_action(state)
                action = self.actor.predict(state)[0]
                # action = self.online_logus.choose_action(state)
                # if ep == 0:
                # self.eval_env.render()

                next_state, reward, terminated, truncated, info = self.eval_env.step(action)
                avg_reward += reward
                state = next_state
                done = terminated or truncated
            # self.eval_env.close()

        avg_reward /= n_episodes
        return avg_reward

    @property
    def _evec_values(self):
        if isinstance(self.env.observation_space, gym.spaces.Discrete):
            nS = self.env.observation_space.n
            nA = self.env.action_space.n
            logu = np.zeros((nS, nA))
            for s in range(nS):
                s_dev = torch.FloatTensor([s]).to(self.device)
                for a in range(nA):
                    with torch.no_grad():
                        logu_ = self.online_logu(s_dev).cpu().squeeze(0)
                        logu[s, a] = logu_[a].numpy()
        else:
            raise NotImplementedError(
                "Can only provide left e.v. for discrete state spaces.")
        return np.exp(logu)


def main():
    env_id = 'CartPole-v1'
    # env_id = 'Taxi-v3'
    # env_id = 'CliffWalking-v0'
    # env_id = 'Acrobot-v1'
    # env_id = 'LunarLander-v2'
    # env_id = 'Pong-v'
    # env_id = 'FrozenLake-v1'
    # env_id = 'MountainCar-v0'
    env_id = 'Pendulum-v1'
    # env_id = 'Ant-v4'
    from darer.hparams import mcar_hparams as config
    agent = LogUActor(env_id, **config, device='cpu', log_dir=None, num_nets=2)
    agent.learn(total_timesteps=50_000_000)


if __name__ == '__main__':
    for _ in range(30):
        main()
