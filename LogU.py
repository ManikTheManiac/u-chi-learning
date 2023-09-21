from disc_envs import get_environment
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
import os
from torch.nn import functional as F
from frozen_lake_env import MAPS, ModifiedFrozenLake, generate_random_map
from visualization import plot_dist
from stable_baselines3.common.utils import polyak_update
from gym.wrappers import TimeLimit


class Memory(object):
    def __init__(self, memory_size: int, device='cpu') -> None:
        self.memory_size = memory_size
        self.device = device
        self.buffer = deque(maxlen=self.memory_size)

    def add(self, experience) -> None:
        self.buffer.append(experience)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size: int, continuous: bool = True):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if continuous:
            rand = random.randint(0, len(self.buffer) - batch_size)
            batch = [self.buffer[i] for i in range(rand, rand + batch_size)]
        else:
            indexes = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
            batch = [self.buffer[i] for i in indexes]
        
        states, next_states, actions, next_actions, rewards, dones = zip(*batch)
        # Now convert to tensors:
        states = torch.from_numpy(np.array(states, dtype=np.float32)).to(self.device)
        next_states = torch.from_numpy(np.array(next_states, dtype=np.float32)).to(self.device)
        actions = torch.from_numpy(np.array(actions, dtype=np.float32)).to(self.device)
        next_actions = torch.from_numpy(np.array(next_actions, dtype=np.float32)).to(self.device)
        rewards = torch.from_numpy(np.array(rewards, dtype=np.float32)).to(self.device)
        dones = torch.from_numpy(np.array(dones, dtype=np.float32)).to(self.device)

        return states, next_states, actions, next_actions, rewards, dones

    def clear(self):
        self.buffer.clear()


class LogUNet(nn.Module):
    def __init__(self, env, device='cuda', hidden_dim=256):
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
        # Check if in batch mode:
        if isinstance(x, int):
            x = torch.from_numpy(np.array([x])).to(device=self.device)
        else:
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).to(device=self.device)

        x = preprocess_obs(x, self.env.observation_space)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x

    def get_chi(self, logu_a):
        prior_policy = 1 / self.nA
        chi = torch.sum(prior_policy * torch.exp(logu_a))
        return chi
        
    def choose_action(self, state, greedy=False):
        with torch.no_grad():
            logu = self.forward(state)

            if greedy:
                # raise NotImplementedError("Greedy not implemented (possible bug?).")
                a = logu.argmax()
                return a.item()

            # First subtract a baseline:
            logu = logu - (torch.max(logu) + torch.min(logu))/2
            dist = torch.exp(logu) * 1 / self.nA
            dist = dist / torch.sum(dist)
            c = Categorical(dist)
            a = c.sample()

        return a.item()


class LogULearner:
    def __init__(self, 
                 env,
                 beta,
                 learning_rate,
                 batch_size,
                 buffer_size,
                 target_update_interval,
                 tau,
                 hidden_dim=64,
                 tau_theta=0.001,
                 gradient_steps=1,
                 device='cpu',
                 run_name='',
                 log_dir=None,
                 log_interval=1000,
                 save_checkpoints = False,
                 ) -> None:
        self.env = env
        # make another instance for evaluation purposes only:
        self.eval_env = env
        self._vec_normalize_env = unwrap_vec_normalize(env)
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
        
        self.replay_buffer = Memory(buffer_size, device=device)
        self.ref_action = None
        self.ref_state = None
        self.ref_reward = None
        self.theta = torch.Tensor([-1]).to(self.device)
        self.eval_auc = 0
        # self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Set up the logger:
        if log_dir is None:
            name = str(len(os.listdir('tmp/uchi'))) if run_name == '' else run_name
            tmp_path = "tmp/uchi/run_" + name
        else:
            tmp_path = log_dir
        os.makedirs(tmp_path, exist_ok=True)
        self.logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

        self._n_updates = 0
        self.env_steps = 0
        self._initialize_networks()

    def _initialize_networks(self):

        self.online_logu = LogUNet(self.env, hidden_dim=self.hidden_dim, device=self.device)
        self.target_logu = LogUNet(self.env, hidden_dim=self.hidden_dim, device=self.device)
        self.target_logu.load_state_dict(self.online_logu.state_dict())

        # Make LogU learnable:
        self.optimizer = torch.optim.Adam(self.online_logu.parameters(), lr=self.learning_rate)


    def learn(self,):
        # replay = self.replay_buffer.sample(self.batch_size, env=self._vec_normalize_env)
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
            # next_chi = self.target_logu.get_chi(self.target_logu(next_states).gather(1, next_actions.long()))
            with torch.no_grad():
                ref_action = torch.from_numpy(np.array(self.ref_action, dtype=np.float32)).to(self.device).unsqueeze(0)
                ref_clogu = self.online_logu(self.ref_state).squeeze(0)[self.ref_action]#.gather(1, ref_action.long())

                ref_logu = self.target_logu(self.ref_next_state)
                ref_chi = self.target_logu.get_chi(ref_logu)

                target_next_logu = self.target_logu(next_states)
                next_logu = target_next_logu.gather(1, next_actions.long())
                next_chi = self.target_logu.get_chi(next_logu)
                # new_theta = torch.mean(rewards - 1/self.beta * (curr_logu - torch.log(next_chi)))
                # new_theta = torch.mean(rewards - torch.log(next_chi))
                new_theta = self.ref_reward - torch.log(ref_chi)
                # new_theta = torch.Tensor([-0.9975]).to(self.device) #9791321771142604

                expected_curr_logu = self.beta * (rewards + self.theta) + \
                      (1 - dones) * next_logu#next_chi)# / ref_chi)
                # self.theta = torch.clamp(new_theta, 0, 1)
                self.theta = (1 - self.tau_theta)*self.theta + self.tau_theta*new_theta

                
            self.logger.record("theta", self.theta.item())
            self.logger.record("avg logu", curr_logu.mean().item())
            # Huber loss:
            loss = F.smooth_l1_loss(curr_logu, expected_curr_logu)
            self.logger.record("loss", loss.item())
            self.optimizer.zero_grad()
            # Increase update counter
            self._n_updates += self.gradient_steps
            
            # Clip gradient norm
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.online_logu.parameters(), 10)

            # Log the average gradient:
            with torch.no_grad():
                ps = []
                for p in self.online_logu.parameters():
                    if p.grad is not None:
                        # ensure not nan:
                        if not torch.isnan(p.grad.mean()):
                            ps.append(p.grad.mean().item())

            self.logger.record("avg_grad", np.mean(ps))
            self.optimizer.step()
    
    def learn_online(self, total_timesteps):
        while self.env_steps < total_timesteps:
        # interact with the environment, episodically:
            state = self.env.reset()
            if self.env_steps == 0:
                self.ref_state = state
            episode_reward = 0
            done = False
            while not done:
                # self.beta = min(25, self.beta * 1.0001)
                if isinstance(self.env.observation_space, gym.spaces.Discrete):
                    state = np.array([state])

                torch_state = torch.FloatTensor(state).to(self.device)
                # if action is None:
                action = self.online_logu.choose_action(torch_state)
                # take a random action:
                # action = self.env.action_space.sample()
                next_state, reward, done, _ = self.env.step(action)

                if self.env_steps == 0:
                    self.ref_action = action
                    self.ref_reward = reward
                    self.ref_next_state = next_state

                # if self.env_steps % 30 == 0: # should this be changed to only learn after done = True?
                if done:
                    if self.replay_buffer.size() > self.batch_size: # or learning_starts?
                        # Begin learning:
                        self.learn()
                if self.env_steps % self.target_update_interval == 0:
                    # Do a Polyak update of parameters:
                    polyak_update(self.online_logu.parameters(), self.target_logu.parameters(), self.tau)

        
                self.env_steps += 1
                next_action = self.online_logu.choose_action(next_state)
                episode_reward += reward
                self.replay_buffer.add((state, next_state, action, next_action, reward, done))
                state = next_state
                # action = next_action
        
                if self.env_steps % self.log_interval == 0:
                    avg_eval_rwd = self.evaluate()
                    self.eval_auc += avg_eval_rwd
                    if self.save_checkpoints:
                        torch.save(self.online_logu.state_dict(), 'sql-policy.para')
                    self.logger.record("Env. steps:", self.env_steps)
                    self.logger.record("Eval. reward:", avg_eval_rwd)
                    self.logger.record("eval_auc", self.eval_auc)
                    self.logger.record("Beta:", self.beta)
                    self.logger.dump(step=self.env_steps)

                # if done:
                #     print("solved in rollout!")


    def evaluate(self, n_episodes=1):
        # run the current policy and return the average reward
        avg_reward = 0.
        # Wrap a timelimit:
        # self.eval_env = TimeLimit(self.eval_env, max_episode_steps=500)
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
                # print(reward)
        avg_reward /= n_episodes
        self.eval_env.close()
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
                        logu[s,a] = logu_[a].numpy()
        else:
            raise NotImplementedError("Can only provide left e.v. for discrete state spaces.")
        return np.exp(logu)

def main():
    desc = generate_random_map(size=4)
    env = gym.make('FrozenLake-v1', is_slippery=0)#, desc=desc)
    # env = get_environment('Pendulum', nbins=5, max_episode_steps=200)
    env = gym.make('CartPole-v1')
    # env = gym.make('MountainCar-v0')
    # use a 500 timestep limit:

    # env = TimeLimit(env.env, max_episode_steps=500)
    # env = gym.make('FrozenLake-v1', is_slippery=False)#, desc=desc)
    # n_action = 4
    # max_steps = 200
    # desc = np.array(MAPS['3x5uturn'], dtype='c')
    # env_src = ModifiedFrozenLake(
    #     n_action=n_action, max_reward=1, min_reward=0,
    #     step_penalization=0, desc=desc, never_done=False, cyclic_mode=True,
    #     # between 0. and 1., a probability of staying at goal state
    #     # an integer. 0: deterministic dynamics. 1: stochastic dynamics.
    #     slippery=0,
    # )
    # env = TimeLimit(env_src, max_episode_steps=max_steps)
    agent = LogULearner(env.env, beta=3.8, learning_rate=4e-3, batch_size=450, buffer_size=200000, 
                        target_update_interval=6000, device='cpu', gradient_steps=7, tau_theta=0.0000413, tau=0.88,#0.001, 
                        log_interval=1000, hidden_dim=256)
    agent.learn_online(5000_000)
    print(f'Theta: {agent.theta}')
    print(agent._evec_values)
    pi = agent._evec_values.reshape((16,4))
    pi /= np.sum(pi, axis=1, keepdims=True)
    desc = np.array(desc, dtype='c')
    plot_dist(desc, pi, titles=['LogU'], filename='logu.png')


if __name__ == '__main__':
    for _ in range(1):
        main()