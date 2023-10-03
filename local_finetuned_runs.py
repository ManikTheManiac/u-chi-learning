import argparse
from darer.MultiLogU import LogULearner
from CustomDQN import CustomDQN
from LogURawlik import LogULearner
from CustomPPO import CustomPPO
# from LogU import LogULearner
from MultiLogU import LogULearner
from hparams import cartpole_hparams0, cartpole_dqn, cartpole_rawlik, cartpole_ppo, mcar_hparams
import time

env = 'CartPole-v1'
# env = 'MountainCar-v0'
configs = {'logu': cartpole_hparams0, 'dqn': cartpole_dqn}


def runner(algo):
    if algo == 'logu':
        if env == 'CartPole-v1':
            config = cartpole_hparams0
        elif env == 'MountainCar-v0':
            config = mcar_hparams
        # config = cartpole_hparams0
        algo = LogULearner
    elif algo == 'dqn':
        config = cartpole_dqn
        algo = CustomDQN
    elif algo == 'ppo':
        config = cartpole_ppo
        algo = CustomPPO

    model = algo(env, **config, log_dir='ft/cartpole',
                 device='cpu', log_interval=500, num_nets=1)
    model.learn(total_timesteps=5_000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--count', type=int, default=1)
    parser.add_argument('-a', '--algo', type=str, default='logu')
    args = parser.parse_args()

    start = time.time()
    for i in range(args.count):
        runner(args.algo)
        print(f"Finished run {i+1}/{args.count}")
    print(f"trained in {time.time() - start}")
