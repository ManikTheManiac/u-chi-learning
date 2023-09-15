# From https://github.com/argearriojas/UAI23-Arriojas_611/blob/main/classic_control_discrete_latest.py
import numpy as np
from gym.envs.classic_control import CartPoleEnv, MountainCarEnv, PendulumEnv, AcrobotEnv
from gym.wrappers import TimeLimit, TransformReward
from gym import spaces, ActionWrapper, Env


class ExtendedCartPoleEnv(CartPoleEnv):
    def __init__(self):
        self.metadata["render_fps"] = 60
        super().__init__()

    def step(self, action):
        next_state, reward, terminated, truncated, info = super().step(action)

        if self.steps_beyond_terminated == 0:
            # need to make this an instantaneus reward drop when terminated
            reward = 0.

        return next_state, reward, terminated, truncated, info


class ExtendedMountainCarEnv(MountainCarEnv):
    def __init__(self, goal_velocity=0):
        self.metadata["render_fps"] = 60
        super().__init__(goal_velocity=goal_velocity)

    def step(self, action):
        next_state, reward, terminated, truncated, info = super().step(action)

        if self.state[0] >= self.goal_position:
            # need to make this an instantaneus reward when terminated
            reward = 0.
        return next_state, reward, terminated, truncated, info


class ExtendedPendulum(PendulumEnv):
    def __init__(self, g=10):
        self.metadata["render_fps"] = 60
        super().__init__(g=g)
        high = np.array([np.pi, self.max_speed], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

    # have to deal with non-standard observation manipulation
    def _get_obs(self):
        th, thdot = self.state

        # make the angle periodic
        th = (th + np.pi) % (2 * np.pi) - np.pi

        return np.array([th, thdot], dtype=np.float32)


class ExtendedAcrobot(AcrobotEnv):
    def __init__(self):
        self.metadata["render_fps"] = 60
        super().__init__()
        high = np.array([np.pi, np.pi, self.MAX_VEL_1, self.MAX_VEL_2], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

    def _get_ob(self):
        th1, th2, th1dot, th2dot = self.state
        th1 = (th1 + np.pi) % (2 * np.pi) - np.pi
        th2 = (th2 + np.pi) % (2 * np.pi) - np.pi

        return np.array([th1, th2, th1dot, th2dot], dtype=np.float32)



class DiscretizeAction(ActionWrapper):
    def __init__(self, env: Env, nbins: int) -> None:
        super().__init__(env)

        assert isinstance(env.action_space, spaces.Box)
        assert len(env.action_space.shape) == 1

        self.ndim_actions, = env.action_space.shape
        self.powers = [nbins ** (i-1) for i in range(self.ndim_actions, 0, -1)]

        low = env.action_space.low
        high = env.action_space.high
        self.action_mapping = np.linspace(low, high, nbins)
        self.action_space = spaces.Discrete(nbins ** self.ndim_actions)
    
    def action(self, action):
        
        a = action
        unwrapped_action = np.zeros((self.ndim_actions,), dtype=float)

        for i, p in enumerate(self.powers):
            idx, a = a // p, a % p
            unwrapped_action[i] = self.action_mapping[idx, i]

        return unwrapped_action
    

def get_environment(env_name, nbins, max_episode_steps=0, reward_offset=0):

    if env_name == 'CartPole':
        env = ExtendedCartPoleEnv()
    elif env_name == 'MountainCar':
        env = ExtendedMountainCarEnv()
    elif env_name == 'Pendulum':
        env = ExtendedPendulum()
        env = DiscretizeAction(env, nbins=3)
    elif env_name == 'Pendulum5':
        env = ExtendedPendulum()
        env = DiscretizeAction(env, nbins=5)
    elif env_name == 'Pendulum21':
        env = ExtendedPendulum()
        env = DiscretizeAction(env, nbins=21)
    elif env_name == 'Acrobot':
        env = ExtendedAcrobot()
    else:
        raise ValueError(f'wrong environment name {env_name}')

    if reward_offset != 0:
        env = TransformReward(env, lambda r: r + reward_offset)
    if max_episode_steps > 0:
        env = TimeLimit(env, max_episode_steps)

    return env