import numpy as np

from utils import chi

class MF_uchi:
    def __init__(self, env, beta=1, u_ref_state=None, prior_policy=None) -> None:

        self.env = env
        self.beta = beta
        if u_ref_state is None:
            self.u_ref_state = (0, 0)
        else:
            self.u_ref_state = u_ref_state

        if prior_policy is None:
            self.prior_policy = np.ones((env.nS, env.nA)) / env.nA
        else:
            self.prior_policy = prior_policy

        # Start the env in provided reference (initial state-action pair).
        # Take a step to find the reference reward and the chi reference state (next state)
        self.env.reset()
        self.env.s, action = self.u_ref_state
        self.chi_ref_state, self.reference_reward, _, _ = self.env.step(action)

        self.init_u = np.ones((self.env.nS, self.env.nA)) 
        self.init_chi = chi(self.init_u, self.env.nS, self.env.nA, self.prior_policy)
        self.ref_chi = self.init_chi[self.chi_ref_state]

        # We will be updating log(u) and chi on each iteration
        self.logu = np.log(self.init_u)
        self.chi = self.init_chi


    def train(self, sarsa_experience, alpha, beta) -> np.ndarray:

        for (state, action, reward, next_state, next_action), _ in sarsa_experience:
            loguold = self.logu

            delta_reward = reward - self.reference_reward

            # Update log(u) based on the u-chi relationship
            self.logu[state,action] = (beta * delta_reward + np.log(self.chi[next_state]/self.ref_chi) )      
            
            # Learn logu update    
            self.logu = loguold * (1 - alpha) + alpha * self.logu
            self.logu -= self.logu[self.u_ref_state]

            self.chi = chi(np.exp(self.logu), self.env.nS, self.env.nA, self.prior_policy)

        return self.logu

    @property
    def policy(self):
        pi = self.prior_policy * np.exp(self.logu)
        pi /= pi.sum(axis=1, keepdims=True)
        return pi

    @property
    def theta(self):
        return -self.beta * self.reference_reward - np.log(self.chi[self.chi_ref_state])

