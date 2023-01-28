"""
Model-free algorithm, uses experience from (fixed) prior policy.
Uses a neural network for function approximation of u and chi.
"""
from agents import MF_uchi
import matplotlib.pyplot as plt
import numpy as np
from frozen_lake_env import ModifiedFrozenLake, MAPS
from gym.wrappers import TimeLimit
from utils import gather_experience, get_dynamics_and_rewards, printf, solve_unconstrained
from visualization import make_animation, plot_dist
from NNUChi import NNUChi
import wandb
from wandb.integration.sb3 import WandbCallback
import gym
# Assuming deterministic dynamics only for now:

beta = 4
n_action = 4
max_steps = 200
desc = np.array(MAPS['7x7holes'], dtype='c')
env_src = ModifiedFrozenLake(
    n_action=n_action, max_reward=-0, min_reward=-1,
    step_penalization=1, desc=desc, never_done=False, cyclic_mode=True,
    # between 0. and 1., a probability of staying at goal state
    # an integer. 0: deterministic dynamics. 1: stochastic dynamics.
    slippery=0,
)
# env = TimeLimit(env_src, max_episode_steps=max_steps)
env = env_src
# NT: No touch (only use for comparison to ground truth)
dynamics_NT, rewards_NT = get_dynamics_and_rewards(env)
n_states = env.nS
n_actions = env.nA
prior_policy = np.ones((n_states, n_actions)) / n_actions

solution = solve_unconstrained(
    beta, dynamics_NT, rewards_NT, prior_policy, eig_max_it=1_000_000, tolerance=1e-12)
l_true, u_true, v_true, optimal_policy, optimal_dynamics, estimated_distribution = solution

# Must "wisely" choose a reference state to normalize u by.
# results = dict(step=[], theta=[], kl=[])
env_name = "FrozenLake-v1"
env = gym.make(env_name, is_slippery=False)

model = NNUChi(env, beta=10, u_ref_state=(0, 0))

with wandb.init(
    project="LogU-Chi",
    config={
        "env_name": "FrozenLake-v1",
        "greedy": True,
    },
    sync_tensorboard=True
) as run:
    # cb = WandbCallback(
    #     gradient_save_freq=500,
    #     model_save_path=f"models/uchi/{run.id}",
    #     verbose=2,
    # )
    model.learn(total_timesteps=1000000)


# plt.figure()
# plt.title("Policy Error")
# plt.plot(results['kl'])
# plt.show()

# u_true = u_true.A
# u_true = u_true.reshape(env.nS, env.nA)
# plt.figure()
# plt.title("Learned vs. True Left Eigenvector")
# plt.plot(u_true.flatten(), label='True')
# u_est = np.exp(agent.logu).flatten()
# # rescale
# u_est = u_est * (u_true.max() / u_est.max())
# plt.plot(u_est, label='Learned')
# plt.legend()
# plt.show()

# pi_learned = agent.policy
# plot_dist(env.desc, pi_learned, optimal_policy,
#           titles=["Learned policy", "True policy"])
# print(-np.log(l_true))
# print(agent.theta)
# plt.figure()
# plt.title('Learned vs. True Eigenvalue')
# plt.plot(thetas, label='Learned')
# plt.hlines(-np.log(l_true), 0, n_iteration, linestyles='dashed', label='True')
# plt.show()
