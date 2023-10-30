from stable_baselines3 import SAC
import gymnasium as gym
# run sac on half cheetah:

env = gym.make('HalfCheetah-v4')
model = SAC('MlpPolicy', env, learning_starts=10_000, verbose=4)
model.learn(total_timesteps=100_000)

