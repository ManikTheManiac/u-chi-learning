import numpy as np
from copy import deepcopy
from utils import chi


class MF_uchi:
    def __init__(self, env, beta=1, u_ref_state=None, prior_policy=None, stochastic=False) -> None:

        self.env = env
        self.env_copy = deepcopy(env)
        self.beta = beta
        self.stochastic = stochastic

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
        # if stochastic:
        #     n_samples=10
        #     self.chi_ref_state = []
        #     self.reference_reward = 0
        #     for i in range(n_samples):
        #         self.env.reset()
        #         self.env.s, action = self.u_ref_state
        #         chi_ref, ref_rwd, _, _ = self.env.step(action)
        #         self.chi_ref_state.append(chi_ref)
        #         self.reference_reward += ref_rwd
        #     self.reference_reward /= n_samples
            
        # else:
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


            # Update log(u) based on the u-chi relationship
            if not self.stochastic:
                delta_reward = reward - self.reference_reward
                self.logu[state,action] = (beta * delta_reward + np.log(self.chi[next_state]/self.ref_chi) )     
            
            if self.stochastic:
                raise NotImplementedError
                # We have to sample from the next available reference state and corresponding reward
                # We make a copy of the environment, start it in the reference state, and take a step
                # We then update the logu and chi
                sample_sum = 0
                num_samples = 10
                for i in range(num_samples):
                    self.env_copy.reset()
                    self.env_copy.s, action = self.u_ref_state
                    chi_ref_state, reference_reward, _, _ = self.env_copy.step(action)
                    delta_reward = reward - reference_reward
                    sample_sum += (beta * delta_reward + np.log(self.chi[next_state]/self.chi[chi_ref_state]) )     

                self.logu[state,action] = sample_sum / num_samples
            
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

###

class NN_uchi:
    def __init__(self, env, beta=1, u_ref_state=None, prior_policy=None, stochastic=False) -> None:

        self.env = env
        self.env_copy = deepcopy(env)
        self.beta = beta
        self.stochastic = stochastic

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
        # if stochastic:
        #     n_samples=10
        #     self.chi_ref_state = []
        #     self.reference_reward = 0
        #     for i in range(n_samples):
        #         self.env.reset()
        #         self.env.s, action = self.u_ref_state
        #         chi_ref, ref_rwd, _, _ = self.env.step(action)
        #         self.chi_ref_state.append(chi_ref)
        #         self.reference_reward += ref_rwd
        #     self.reference_reward /= n_samples
            
        # else:
        self.chi_ref_state, self.reference_reward, _, _ = self.env.step(action)

        self.init_u = np.ones((self.env.nS, self.env.nA)) 
        self.init_chi = chi(self.init_u, self.env.nS, self.env.nA, self.prior_policy)
        self.ref_chi = self.init_chi[self.chi_ref_state]

        # We will be updating log(u) and chi on each iteration
        self.logu = np.log(self.init_u)
        self.chi = self.init_chi

        self.network_setup()

    def train(self, sarsa_experience, alpha, beta) -> np.ndarray:

        for (state, action, reward, next_state, next_action), _ in sarsa_experience:
            loguold = self.logu


            # Update log(u) based on the u-chi relationship
            if not self.stochastic:
                delta_reward = reward - self.reference_reward
                self.logu[state,action] = (beta * delta_reward + np.log(self.chi[next_state]/self.ref_chi) )     
            
            if self.stochastic:
                raise NotImplementedError
                # We have to sample from the next available reference state and corresponding reward
                # We make a copy of the environment, start it in the reference state, and take a step
                # We then update the logu and chi
                sample_sum = 0
                num_samples = 10
                for i in range(num_samples):
                    self.env_copy.reset()
                    self.env_copy.s, action = self.u_ref_state
                    chi_ref_state, reference_reward, _, _ = self.env_copy.step(action)
                    delta_reward = reward - reference_reward
                    sample_sum += (beta * delta_reward + np.log(self.chi[next_state]/self.chi[chi_ref_state]) )     

                self.logu[state,action] = sample_sum / num_samples
            
            # Learn logu update    
            self.logu = loguold * (1 - alpha) + alpha * self.logu
            self.logu -= self.logu[self.u_ref_state]
                    # ft the model on the training dataset
            # let's do a one-hot encoding on the state-action pair
            x = np.zeros(self.env.nS*self.env.nA)
            x[state*self.env.nA + action] = 1
            y=self.logu.flatten().reshape(196,)
            print(y.shape)
            
            self.logu_model.fit(x, y, epochs=1, batch_size=1, verbose=1)

            ch_true=chi(np.exp(self.logu), self.env.nS, self.env.nA, self.prior_policy)
            x_chi = np.zeros(self.env.nS)
            x_chi[state] = 1
            y_chi = ch_true.flatten().reshape(49,)
            self.chi_model.fit(x_chi, y_chi, epochs=1, batch_size=1, verbose=1)

        return self.logu

    def network_setup(self):
        # from sklearn.metrics import mean_squared_error
        from keras.models import Sequential
        from keras.layers import Dense
        # design the neural network model
        logu_model = Sequential()
        logu_model.add(Dense(196, input_shape=(1, ), activation='relu', kernel_initializer='he_uniform'))
        # logu_model.add(Dense(10, input_dim=self.env.nS, activation='relu', kernel_initializer='he_uniform'))
        # logu_model.add(Dense(1, input_dim=self.env.nS*self.env.nA))
        # define the loss function and optimization algorithm
        logu_model.compile(loss='mse', optimizer='adam')
        
        self.logu_model = logu_model

        chi_model = Sequential()
        chi_model.add(Dense(49, input_shape=(1,), activation='relu'))
        chi_model.compile(loss='mse', optimizer='adam')
        
        self.chi_model = chi_model

    def network_predict_logu(self, S, A):
        
        yhat = self.logu_model.predict(S*A)
        # # inverse transforms
        # x_plot = scale_x.inverse_transform(x)
        # y_plot = scale_y.inverse_transform(y)
        # yhat_plot = scale_y.inverse_transform(yhat)

        return yhat

    @property
    def policy(self):
        pi = self.prior_policy * np.exp(self.logu)
        pi /= pi.sum(axis=1, keepdims=True)
        return pi

    @property
    def theta(self):
        return -self.beta * self.reference_reward - np.log(self.chi[self.chi_ref_state])

