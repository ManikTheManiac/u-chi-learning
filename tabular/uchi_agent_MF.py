"""
Model-free algorithm, uses experience collected in batches.
"""

from agents import MF_uchi
import matplotlib.pyplot as plt
import numpy as np
from frozen_lake_env import ModifiedFrozenLake, MAPS
from gym.wrappers import TimeLimit
from utils import chi, gather_experience, get_dynamics_and_rewards, printf, solve_unconstrained
from visualization import make_animation, plot_dist, save_plots

# Assuming deterministic dynamics only for now:

beta = 8
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
env = TimeLimit(env_src, max_episode_steps=max_steps)

# NT: No touch (only use for comparison to ground truth)
dynamics_NT, rewards_NT = get_dynamics_and_rewards(env)
n_states = env.nS
n_actions = env.nA
prior_policy = np.ones((n_states, n_actions)) / n_actions

solution = solve_unconstrained(
    beta, dynamics_NT, rewards_NT, prior_policy, eig_max_it=1_000_000, tolerance=1e-12)
l_true, u_true, v_true, optimal_policy, optimal_dynamics, estimated_distribution = solution

# Must "wisely" choose a reference state to normalize u by.

results = dict(step=[], theta=[], kl=[])

agent = MF_uchi(env, beta=beta, u_ref_state=(1, 0), stochastic=False)
step = 0
max_it = 2000
# alpha_scale = 0.01
# decay = 2e3
batch_size = 100
for it in range(max_it):
    printf('Iteration', it, max_it)
    alpha = 0.0001  # decay / (decay + step) * alpha_scale

    sarsa_experience = gather_experience(
        env, agent.prior_policy, batch_size=batch_size, n_jobs=4)
    agent.train(sarsa_experience, alpha, beta)

    kl = - (agent.policy * (np.log(agent.policy) - np.log(optimal_policy))).sum()
    theta = agent.theta

    step += batch_size

    results['theta'].append(theta)
    results['kl'].append(kl)

print(-np.log(l_true))
print(agent.theta)
save_plots(agent, results, u_true, l_true, name='MF')
