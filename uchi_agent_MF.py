"""
Model-free algorithm, iterates over every initial state and action manually.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from frozen_lake_env import ModifiedFrozenLake, MAPS
from gym.wrappers import TimeLimit
from utils import gather_experience, get_dynamics_and_rewards, printf, solve_unconstrained
from visualization import make_animation, plot_dist

# Assuming deterministic dynamics only for now:

beta = 6
n_action = 4
max_steps = 200
desc = np.array(MAPS['5x4uturn'], dtype='c')
env_src = ModifiedFrozenLake(
    n_action=n_action, max_reward=-0, min_reward=-1,
    step_penalization=1, desc=desc, never_done=False, cyclic_mode=True,
     # between 0. and 1., a probability of staying at goal state
    slippery=0, # an integer. 0: deterministic dynamics. 1: stochastic dynamics.
)
env = TimeLimit(env_src, max_episode_steps=max_steps)

# NT: No touch (only use for comparison to ground truth)
dynamics_NT, rewards_NT = get_dynamics_and_rewards(env)
n_states = env.nS
n_actions = env.nA
prior_policy = np.ones((n_states, n_actions)) / n_actions

solution = solve_unconstrained(beta, dynamics_NT, rewards_NT, prior_policy, eig_max_it=1_000_000, tolerance=1e-12)
l_true, u_true, v_true, optimal_policy, optimal_dynamics, estimated_distribution = solution

# Must "wisely" choose a reference state to normalize u by.

from agents import MF_uchi
results = dict(step=[], theta=[], kl=[])

agent = MF_uchi(env, beta=beta, u_ref_state=(1,1))
step = 0
max_it = 50
alpha_scale = 0.01
decay = 2e3
batch_size = 50
for it in range(max_it):
    printf('Iteration', it, max_it)
    alpha = 0.05#decay / (decay + step) * alpha_scale

    sarsa_experience = gather_experience(env, agent.prior_policy, batch_size=batch_size, n_jobs=4)
    agent.train(sarsa_experience, alpha, beta)

    kl = - (agent.policy * (np.log(agent.policy) - np.log(optimal_policy))).sum()
    theta = agent.theta

    step += batch_size

    results['theta'].append(theta)
    results['kl'].append(kl)

# plt.figure()
# plt.title("Dominant Eigenvalue")
# plt.plot(thetas)
# plt.show()

plt.figure()
plt.title("Policy Error")
plt.plot(results['kl'])
plt.show()

u_true = u_true.A
u_true = u_true.reshape(env.nS, env.nA)
plt.figure()
plt.title("Learned vs. True Left Eigenvector")
plt.plot(u_true.flatten(), label='True')
u_est = np.exp(agent.logu).flatten()
# rescale
u_est = u_est * (u_true.max() / u_est.max())
plt.plot(u_est, label='Learned')
plt.legend()
plt.show()

pi_learned = agent.policy
plot_dist(env.desc, pi_learned, optimal_policy, titles=["Learned policy", "True policy"])
print(-np.log(l_true))
print(agent.theta)
# plt.figure()
# plt.title('Learned vs. True Eigenvalue')
# plt.plot(thetas, label='Learned')
# plt.hlines(-np.log(l_true), 0, n_iteration, linestyles='dashed', label='True')
# plt.show()