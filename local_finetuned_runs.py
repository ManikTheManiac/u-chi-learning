import argparse
import wandb
import gym
from LogU import LogULearner
from hparams import cartpole_hparams

# env = 'MountainCar-v0'# = gym.make('CartPole-v1')
env = 'CartPole-v1'
config = cartpole_hparams

def runner(config=config):
    config['hidden_dim'] = args.hidden_dim
    model = LogULearner(env, **config, log_dir='ft/cartpole3', device='cpu', log_interval=500)
    model.learn_online(total_timesteps=150_000)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, default=256)
    parser.add_argument("-c", "--count", type=int, default=40)
    
    args = parser.parse_args()

    for i in range(args.count):
        runner()