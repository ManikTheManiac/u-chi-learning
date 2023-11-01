import gym
import torch
import numpy as np
from stable_baselines3.common.vec_env import unwrap_vec_normalize
from stable_baselines3.common.logger import configure
import os
from torch.nn import functional as F
from stable_baselines3.common.utils import polyak_update
from gym.wrappers import TimeLimit
import time
from ReplayBuffers import Memory
from Models import UNet


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
                 device='cpu',
                 run_name='',
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
        
        self.replay_buffer = Memory(buffer_size, device=device)
        self.ref_action = None
        self.ref_state = None
        self.ref_reward = None
        self.theta = torch.Tensor([0]).to(self.device)
        self.eval_auc = 0
        self.num_episodes = 0
        # self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Set up the logger:
        if log_dir is not None:
            run_name = str(len(os.listdir(log_dir)))
            tmp_path = f"{log_dir}_{run_name}"
        
            os.makedirs(tmp_path, exist_ok=True)
            self.logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
        else:
            # print the logs to stdout:
            self.logger = configure(format_strings=["stdout", "csv", "tensorboard"])

        self._n_updates = 0
        self.env_steps = 0
        self._initialize_networks()

    def _initialize_networks(self):
        self.online_u = UNet(self.env, hidden_dim=self.hidden_dim, device=self.device)
        self.target_u = UNet(self.env, hidden_dim=self.hidden_dim, device=self.device)
        self.target_u.load_state_dict(self.online_u.state_dict())

        # Make LogU learnable:
        self.optimizer = torch.optim.Adam(self.online_u.parameters(), lr=self.learning_rate)


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

            curr_u = self.online_u(states)
            curr_u = curr_u.squeeze(1)#self.online_u(states).squeeze(1)
            curr_u = curr_u.gather(1, actions.long())

            with torch.no_grad():
                ref_u = self.online_u(self.ref_next_state)
                ref_chi = self.online_u.get_chi(ref_u)
                new_theta = self.ref_reward - torch.log(ref_chi)
                new_thetas.append(new_theta)

                target_next_u = self.target_u(next_states)
                next_u = target_next_u.gather(1, next_actions.long())
                
                expected_curr_u = torch.exp(self.beta * (rewards + self.theta)) * (1 - dones) * next_u

                
            self.logger.record("theta", self.theta.item())
            self.logger.record("avg u", curr_u.mean().item())
            # Huber loss:
            loss = F.smooth_l1_loss(curr_u, expected_curr_u)
            # MSE loss:
            # loss = F.mse_loss(curr_u, expected_curr_u)
            self.logger.record("loss", loss.item())
            self.optimizer.zero_grad()
            # Increase update counter
            self._n_updates += self.gradient_steps
            
            # Clip gradient norm
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.online_u.parameters(), self.max_grad_norm)

            # Log the average gradient:
            # TODO: put this in a parallel process somehow or use dot prods?
            total_norm = torch.max(torch.stack(
                        [p.grad.detach().abs().max() for p in self.online_u.parameters()]
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
            action = self.online_u.choose_action(state)

            self.num_episodes += 1
            self.rollout_reward = 0
            while not done:
                # take a random action:
                # action = self.env.action_space.sample()
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
                    polyak_update(self.online_u.parameters(), self.target_u.parameters(), self.tau)

        
                self.env_steps += 1
                next_action = self.online_u.choose_action(next_state)
                # next_action = self.env.action_space.sample()

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
                        torch.save(self.online_u.state_dict(), 'sql-policy.para')
                    self.logger.record("Env. steps:", self.env_steps)
                    self.logger.record("Eval. reward:", avg_eval_rwd)
                    self.logger.record("eval_auc", self.eval_auc)
                    self.logger.record("# Episodes:", self.num_episodes)
                    self.logger.record("fps", fps)
                    self.logger.dump(step=self.env_steps)
                    t0 = time.thread_time_ns()
                self.logger.record("Rollout reward:", self.rollout_reward)



    def evaluate(self, n_episodes=5):
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
                action = self.online_u.choose_action(state, greedy=True)
                # if ep == 0:
                    # self.env.render()

                next_state, reward, done, _ = self.eval_env.step(action)

                avg_reward += reward
                state = next_state
        avg_reward /= n_episodes
        self.eval_env.close()
        return avg_reward
    
def main():
    env_id = 'CartPole-v1'
    # env_id = 'Acrobot-v1'
    env_id = 'LunarLander-v2'
    # env_id = 'Pong-v'
    # env_id = 'FrozenLake-v1'
    env_id = 'MountainCar-v0'
    # agent = LogULearner(env_id, beta=4, learning_rate=3e-2, batch_size=1500, buffer_size=45000, 
    #                     target_update_interval=150, device='cpu', gradient_steps=40, tau_theta=0.9, tau=0.75,#0.001, 
    #                     log_interval=100, hidden_dim=256)
    from hparams import cartpole_hparams1 as config
    agent = LogULearner(env_id, **config, log_dir='tmp', device='cuda', max_grad_norm=1, log_interval=2000)
    # agent.learn(250_000)
    from stable_baselines3 import PPO
    # agent = PPO('MlpPolicy', env_id, verbose=1, device='cuda', tensorboard_log='tmp')
    agent.learn(total_timesteps=150_000)
    # print(f'Theta: {agent.theta}')
    # print(agent._evec_values)
    # pi = agent._evec_values.reshape((16,4))
    # pi /= np.sum(pi, axis=1, keepdims=True)
    # desc = np.array(desc, dtype='c')
    # plot_dist(desc, pi, titles=['LogU'], filename='u.png')


if __name__ == '__main__':
    for _ in range(1):
        main()