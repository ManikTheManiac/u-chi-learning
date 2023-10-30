import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabular.frozen_lake_env import ModifiedFrozenLake, MAPS
from gymnasium.wrappers import TimeLimit

from tabular.utils import get_dynamics_and_rewards, solve_unconstrained


from darer.MultiLogU import LogULearner
from darer.hparams import *
config = cartpole_hparams2
config.pop('beta')
map_name = '4x4'
def exact_solution(beta, env):

    dynamics, rewards = get_dynamics_and_rewards(env.unwrapped)
    n_states, SA = dynamics.shape
    n_actions = int(SA / n_states)
    prior_policy = np.ones((n_states, n_actions)) / n_actions

    solution = solve_unconstrained(
        beta, dynamics, rewards, prior_policy, eig_max_it=1_000_000, tolerance=1e-12)
    l_true, u_true, v_true, optimal_policy, optimal_dynamics, estimated_distribution = solution

    print(f"l_true: {l_true}")
    return -np.log(l_true) / beta

def FA_solution(beta, env):
    # Use MultiLogU to solve the environment

    agent = LogULearner(env, **config, log_interval=100, num_nets=2, device='cpu', beta=beta, render=1)
    agent.learn(total_timesteps=70_000)
    # convert agent.theta to float
    theta = agent.theta.item()
    return theta

def main():
    # initialize the environment
    n_action = 5
    max_steps = 200
    desc = np.array(MAPS[map_name], dtype='c')
    env_src = ModifiedFrozenLake(
        n_action=n_action, max_reward=0, min_reward=-1,
        step_penalization=1, desc=desc, never_done=True, cyclic_mode=True,
        # between 0. and 1., a probability of staying at goal state
        # an integer. 0: deterministic dynamics. 1: stochastic dynamics.
        slippery=0,
    )
    env = TimeLimit(env_src, max_episode_steps=max_steps)

    # Set the beta values to test
    betas = np.logspace(1, 1, 4)
    betas = [1, 3, 10]

    exact = [exact_solution(beta, env) for beta in betas]
    print(exact)
    FA = [FA_solution(beta, env) for beta in betas]

    # save the data:
    data = pd.DataFrame({'beta': betas, 'exact': exact, 'FA': FA})
    data.to_csv(f'{map_name}tabular_vs_FA.csv', index=False)

    plt.figure()
    plt.plot(betas, exact, 'ko-', label='Exact')
    plt.plot(betas, FA, label='FA')
    plt.legend()
    plt.savefig(f'{map_name}tabular_vs_FA.png')

if __name__ == '__main__':
    main()