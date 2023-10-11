# The most simple continuous control env.
import gymnasium as gym
import numpy as np

class SimpleEnv(gym.Env):
    def __init__(self) -> None:
        super().__init__()
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.state = np.array([0])
        self.reward_range = (-1, 1)

    def step(self, action):
        # Goal is to stay near center: reward = -|x|
        # transition to state + action with periodic? BCs:
        new_state = 1.2*action + self.state
        new_state = np.clip(new_state, -1, 1)

        self.state = new_state
        reward = -np.abs(self.state)**2

        return new_state, reward[0], False, False, {}
    
    def reset(self, seed=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        # Choose a random starting state:
        self.state = np.array([0]) 
        #self.np_random.uniform(low=-1, high=1, size=(1,))
        # self.state = np.array([0])
        return self.state, {}
    
    def render(self, mode='human'):
        pass


    