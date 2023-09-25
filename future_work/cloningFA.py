"""
We want to find the quasi-stationary distribution (QSD) of the Markov chain induced by the prior policy.
Updates will be calculated based on the cloning algorithm, and we will use a simple FF NN to approximate the QSD.
Specifically, we borrow from actor-critic methods and parameterize the QSD as a function of the state, and use a
Gaussian ansatz for the policy.
        # (Output a mean and std for the Gaussian policy)

"""
import gym
import numpy as np
from torch import nn
from torch.nn.functional import relu

class QSD(nn.Module):
    def __init__(self, env, prior_policy=None, hidden_dim=64, batch_size=100):
        super().__init__()
        self.nS = env.nS
        self.nA = env.nA
        self.batch_size = batch_size

        # initialize network:
        self.fc1 = nn.Linear(self.nS * self.nA, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, self.nA)
        self.activation = relu

        # if prior policy is None, use a uniform policy:
        if prior_policy is None:
            prior_policy = np.ones((env.nS, env.nA)) / env.nA

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)        
        x = self.fc3(x)
        x = self.activation(x)
        return x
    
    def absorb(self, population):
        # Query the absorption probabilities:
        abs_probs = self.absorption_model(population)
        # Sample from the absorption probabilities:
        absorb = np.random.choice([True, False], size=len(population), p=abs_probs)
        # Remove the absorbed population:
        population = population[~absorb]
        return population

    def transition(self, population):
        # transition the population according to the dynamics:
        population = self.dynamics(population)
        return population

    def refill(self, population):
        # first calculate required refill size:
        refill_size = self.batch_size - len(population)
        # Check whether all agents were absorbed:
        if refill_size == 0:
            # sample a new population:
            population = self.sample(size=self.batch_size)
            # raise a warning:
            print('Warning: all agents were absorbed.')
            return population
        else:
            
        
    
    def loss(self, y_pred, y_true):
        return nn.MSELoss(y_pred, y_true)

    def evolve(self, population, n_timesteps=1000):
        for t in range(n_timesteps):
            population = self.absorb(population)
            population = self.transition(population)
            population = self.refill(population)
        raise NotImplementedError
    
    def update(self):
        # Sample a batch to generate a population, then evolve it:
        population = self.sample(size=self.batch_size)
        population = self.evolve(population)
        # Learn from the evolved population:
        self.learn(population)
        raise NotImplementedError
        
    
    def learn(self, num_episodes=1000):
        raise NotImplementedError
    


def main():
    env = gym.make('FrozenLake-v1')
    qsd = QSD(env)


if __name__ == "__main__":
    main()
