import gym
import torch
import numpy as np
from stable_baselines3.common.vec_env import unwrap_vec_normalize
import os
from torch.nn import functional as F
from stable_baselines3.common.utils import polyak_update
from gym.wrappers import TimeLimit
import time
from ReplayBuffers import Memory
from Models import LogUNet
from utils import logger_at_folder

class Optimizers():
    def __init__(self, list_of_optimizers: list):
        self.optimizers = list_of_optimizers

    def zero_grad(self):
        for opt in self.optimizers:
            opt.zero_grad()

    def step(self):
        for opt in self.optimizers:
            opt.step()

class TargetNets():
    def __init__(self, list_of_nets):
        self.nets = list_of_nets
    def __iter__(self):
        return iter(self.nets)

    def load_state_dict(self, list_of_state_dicts):
        for new_state, net in zip(list_of_state_dicts, self.nets):
            net.load_state_dict(new_state)

    def polyak(self, new_weights, tau):
        for weights, new_weights in zip(self.parameters(), new_weights):
            polyak_update(new_weights, weights, tau)


    def parameters(self):
        return [net.parameters() for net in self.nets]

class OnlineNets():
    def __init__(self, list_of_nets):
        self.nets = list_of_nets
    def __iter__(self):
        return iter(self.nets)

    def greedy_action(self, state):
        with torch.no_grad():
            logu = torch.stack([net(state) for net in self.nets])
            logu = logu.squeeze(1)
            logu = torch.min(logu, dim=0)[0]
            greedy_action = logu.argmax()
        return greedy_action.item()

    def choose_action(self, state):
        # Get a sample from each net, then sample uniformly over them:
        actions = [net.choose_action(state) for net in self.nets]
        action = np.random.choice(actions)
        # perhaps re-weight this based on pessimism?
        return action.item()
    
    def parameters(self):
        return [net.parameters() for net in self.nets]
    
    def clip_grad_norm(self, max_grad_norm):
        for net in self.nets:
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)


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
                 num_nets=2,
                 tau_theta=0.001,
                 gradient_steps=1,
                 train_freq=-1,
                 max_grad_norm=10,
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
        self.num_nets = num_nets
        
        self.replay_buffer = Memory(buffer_size, device=device)
        self.ref_action = None
        self.ref_state = None
        self.ref_reward = None
        self.theta = torch.Tensor([0]).to(self.device)
        self.eval_auc = 0
        self.num_episodes = 0
        # self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Set up the logger:
        self.logger = logger_at_folder(log_dir, algo_name=f'LogU{num_nets}nets')

        self._n_updates = 0
        self.env_steps = 0
        self._initialize_networks()

    def _initialize_networks(self):
        # self.online_logu = LogUNet(self.env, hidden_dim=self.hidden_dim, device=self.device)
        self.online_logus = OnlineNets(list_of_nets=[LogUNet(self.env, hidden_dim=self.hidden_dim, device=self.device) 
                                for _ in range(self.num_nets)])
        self.target_logus = TargetNets(list_of_nets=[LogUNet(self.env, hidden_dim=self.hidden_dim, device=self.device)
                                for _ in range(self.num_nets)])
        self.target_logus.load_state_dict([logu.state_dict() for logu in self.online_logus])

        # Make (all) LogUs learnable:
        # self.optimizer = torch.optim.Adam(self.online_logu.parameters(), lr=self.learning_rate)
        opts = [torch.optim.Adam(logu.parameters(), lr=self.learning_rate)
                            for logu in self.online_logus]
        self.optimizers = Optimizers(opts)


    def train(self,):
        # replay = self.replay_buffer.sample(self.batch_size, env=self._vec_normalize_env)
        # average self.theta over multiple gradient steps
        new_thetas = torch.zeros(self.gradient_steps)
        for grad_step in range(self.gradient_steps):
            replay = self.replay_buffer.sample(self.batch_size, continuous=False)
            states, next_states, actions, next_actions, rewards, dones = replay

            actions = actions.unsqueeze(1)
            next_actions = next_actions.unsqueeze(1)
            rewards = rewards.unsqueeze(1)
            dones = dones.unsqueeze(1)

            curr_logu = torch.cat([online_logu(states).squeeze(1).gather(1,actions.long())
                                    for online_logu in self.online_logus], dim=1)
            # curr_logu = curr_logu.squeeze(1)#self.online_logu(states).squeeze(1)
            # curr_logu = curr_logu.gather(1, actions.long())
            # # Take min:
            # curr_logu = torch.min(curr_logu, dim=0)[0]

            with torch.no_grad():
                ref_logu = [logu(self.ref_next_state) for logu in self.online_logus]
                ref_chi = torch.stack([logu.get_chi(ref_logu_val) for ref_logu_val, logu in zip(ref_logu, self.online_logus)])
                new_theta = self.ref_reward - torch.log(ref_chi)
                # new_thetas[] (min(new_theta).item())
                new_thetas[grad_step] = torch.min(new_theta)

                # target_next_logu = torch.cat([target_logu(next_states) for target_logu in self.target_logus], dim=1).squeeze(1)
                target_next_logu = torch.cat([target_logu(next_states).gather(1, next_actions.long()) 
                                              for target_logu in self.target_logus],dim=1)
                # next_logu = target_next_logu.gather(1, next_actions.long())
                next_logu = torch.min(target_next_logu, dim=1, keepdim=True)[0]
                # tile next_logu to match curr_logu:
                next_logu = next_logu.repeat(1, self.num_nets)
                
                expected_curr_logu = self.beta * (rewards + self.theta) + (1 - dones) * next_logu

                
            self.logger.record("train/theta", self.theta.item())
            self.logger.record("train/avg logu", curr_logu.mean().item())
            # Huber loss:
            # loss = 0.5* sum([F.smooth_l1_loss(curr_logu, expected_curr_logu) for curr_logu in curr_logus])
            # print(curr_logu.shape, expected_curr_logu.shape)
            loss = F.smooth_l1_loss(curr_logu, expected_curr_logu)
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
        # self.theta = torch.mean(torch.stack(new_thetas))
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
            # Random choice:
            action = self.online_logus.choose_action(state)

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
                    # polyak_update(self.online_logus.nets[0].parameters(), self.target_logus.nets[0].parameters(), self.tau)
                    # polyak_update(self.online_logus.nets[1].parameters(), self.target_logus.nets[1].parameters(), self.tau)

                    self.target_logus.polyak(self.online_logus.parameters(), self.tau)
                    # for target_logu, online_logu in zip(self.target_logus.nets, self.online_logus.nets):
                        # polyak_update(target_logu.parameters(), online_logu.parameters(), self.tau)

        
                self.env_steps += 1
                next_action = self.online_logus.choose_action(next_state)
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
                        torch.save(self.online_logu.state_dict(), 'sql-policy.para')
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
        # Wrap a timelimit:
        # self.eval_env = TimeLimit(self.eval_env, max_episode_steps=500)
        # move to cpu for evaluation:
        # self.eval_env = self.eval_env.to('cpu')
        for ep in range(n_episodes):
            state = self.eval_env.reset()
            done = False
            while not done:
                action = self.online_logus.greedy_action(state)
                # action = self.online_logus.choose_action(state)
                # if ep == 0:
                    # self.env.render()

                next_state, reward, done, _ = self.eval_env.step(action)

                avg_reward += reward
                state = next_state
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
    env_id = 'CartPole-v1'
    # env_id = 'Acrobot-v1'
    # env_id = 'LunarLander-v2'
    # env_id = 'Pong-v'
    # env_id = 'FrozenLake-v1'
    # env_id = 'MountainCar-v0'
    # agent = LogULearner(env_id, beta=4, learning_rate=3e-2, batch_size=1500, buffer_size=45000, 
    #                     target_update_interval=150, device='cpu', gradient_steps=40, tau_theta=0.9, tau=0.75,#0.001, 
    #                     log_interval=100, hidden_dim=256)
    from hparams import cartpole_hparams0 as config
    agent = LogULearner(env_id, **config, device='cuda', log_interval=200, num_nets=2)
    # agent.learn(250_000)
    from stable_baselines3 import PPO
    # agent = PPO('MlpPolicy', env_id, verbose=1, device='cuda', tensorboard_log='tmp')
    agent.learn(total_timesteps=50_000)
    # print(f'Theta: {agent.theta}')
    # print(agent._evec_values)
    # pi = agent._evec_values.reshape((16,4))
    # pi /= np.sum(pi, axis=1, keepdims=True)
    # desc = np.array(desc, dtype='c')
    # plot_dist(desc, pi, titles=['LogU'], filename='logu.png')


if __name__ == '__main__':
    for _ in range(1):
        main()