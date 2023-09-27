import argparse
from LogU import LogULearner
from CustomDQN import CustomDQN
from LogU import LogULearner
from hparams import cartpole_hparams1, cartpole_dqn

env = 'CartPole-v1'
configs = {'logu': cartpole_hparams1, 'dqn': cartpole_dqn}

def runner(algo):
    if algo == 'logu':
        config = cartpole_hparams1
        algo = LogULearner
    elif algo == 'dqn':
        config = cartpole_dqn
        algo = CustomDQN

    model = algo(env, **config, log_dir='ft/benchmark', device='cuda', log_interval=500)
    model.learn(total_timesteps=30_000)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--count', type=int, default=12)
    parser.add_argument('-a', '--algo', type=str, default='logu')
    args = parser.parse_args()

    for i in range(args.count):
        runner('dqn')