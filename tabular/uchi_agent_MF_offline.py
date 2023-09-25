"""
Model-free algorithm, uses previously obtained trajectory data (offline).
"""

raise NotImplementedError

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from frozen_lake_env import ModifiedFrozenLake, MAPS
from gym.wrappers import TimeLimit
from utils import get_dynamics_and_rewards, printf, solve_unconstrained, solve_unconstrained_v1, get_mdp_transition_matrix
from visualization import make_animation, plot_dist
from scipy.sparse import csr_matrix, coo_matrix, lil_matrix
import itertools
    
# Assuming deterministic dynamics only for now:

beta = 5
n_action = 4
max_steps = 200
desc = np.array(MAPS['7x8wall'], dtype='c')
env_src = ModifiedFrozenLake(
    n_action=n_action, max_reward=-0, min_reward=-1,
    step_penalization=1, desc=desc, never_done=False, cyclic_mode=True,
     # between 0. and 1., a probability of staying at goal state
    slippery=0, # an integer. 0: deterministic dynamics. 1: stochastic dynamics.
)
env = TimeLimit(env_src, max_episode_steps=max_steps)


dynamics, rewards = get_dynamics_and_rewards(env)
n_states, SA = dynamics.shape
n_actions = int(SA / n_states)
prior_policy = np.ones((n_states, n_actions)) / n_actions

# twisted_matrix = get_mdp_transition_matrix(dynamics, prior_policy) @ np.diag(rewards.A.flatten())
# X = np.linalg.inv(np.identity(SA) - twisted_matrix)
# print(X.sum(axis=0))


solution = solve_unconstrained(beta, dynamics, rewards, prior_policy, eig_max_it=1_000_000, tolerance=1e-12)
l_true, u_true, v_true, optimal_policy, optimal_dynamics, estimated_distribution = solution
rewards = (np.array(rewards)[0]).reshape(n_states, n_actions)


# Must "wisely" choose a reference state to normalize u by.
u_ref_state = (0,1)
chi_ref_state = int(dynamics[u_ref_state]) # where does the reference state send you to?

delta_rwds = rewards - rewards[u_ref_state]

next_state = dynamics.A[u_ref_state]

def chi(u):
    u = u.reshape(n_states, n_actions)
    return (prior_policy * u).sum(axis=1)


init_u = np.ones((n_states,n_actions)) 
init_chi = chi(init_u)
# init_u = u


# Learning rate and training episodes:
alpha = 0.02
n_iteration = 600

# Track eigenvalue during learning (can use to check convergence):
thetas = []

# We will be updating log(u) and chi on each iteration
logu = np.log(init_u)
ch = init_chi

# Also can track error for convergence:
errs = []

for i in range(n_iteration):
    printf('Iteration', i, n_iteration)
    loguold = logu # Set old logu to current logu (so we can learn and calculate error between the two)
    # Loop over all state action pairs
    for state in range(n_states):
        for action in range(n_actions):
            if (state,action) != u_ref_state:
                # Check which states will be transitioned to: (det. dynamics so only 1 possible next state)

                state_prime = np.argwhere(dynamics.A.T[state*env.nA + action] == 1)[0][0]

                # Update log(u) based on the u-chi relationship
                logu[state,action] = (beta * delta_rwds[state,action] + np.log(ch[state_prime]/ch[chi_ref_state]) )


        

    # Learn logu update    
    logu = loguold * (1 - alpha) + alpha * logu
    logu -= logu[u_ref_state]

    # Update chi at each state-iteration (because it will change each time)
    ch = chi(np.exp(logu))

    # Calculate error between old and new logu
    # chisa = np.array([np.log(ch)]*n_actions).T
    # errs.append(np.abs(logu - (delta_rwds + np.log(chisa))).sum())
    
    errs.append(np.abs(logu.flatten() - np.log(u_true)).sum())

    # Track eigenvalue
    theta = - beta * rewards[u_ref_state] - np.log(ch[chi_ref_state])
    thetas.append(theta)

# plt.figure()
# plt.title("Dominant Eigenvalue")
# plt.plot(thetas)
# plt.show()

plt.figure()
plt.title("logu Error")
plt.plot(errs)
plt.show()

u_true = u_true.A
u_true = u_true.reshape(env.nS, env.nA)
plt.figure()
plt.title("Learned vs. True Left Eigenvector")
plt.plot(u_true.flatten(), label='True')
u_est = np.exp(logu).flatten()
# rescale
u_est = u_est * (u_true.max() / u_est.max())
plt.plot(u_est, label='Learned')
plt.legend()
plt.show()

pi_learned = np.exp(logu)/np.exp(logu).sum(axis=1, keepdims=True)
pi_true = u_true/u_true.sum(axis=1, keepdims=True)
plot_dist(env.desc, pi_learned, pi_true, titles=["Learned policy", "True policy"])
print(l_true)
print(np.exp(-np.mean(thetas[-10:])))
plt.figure()
plt.title('Learned vs. True Eigenvalue')
plt.plot(thetas, label='Learned')
plt.hlines(-np.log(l_true), 0, n_iteration, linestyles='dashed', label='True')
plt.show()