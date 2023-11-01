import argparse
from darer.MultiLogU import LogULearner
from CustomDQN import CustomDQN
from CustomPPO import CustomPPO
# from LogU import LogULearner
from MultiLogU import LogULearner
from hparams import *
import time

# env = 'CartPole-v1'
# env = 'LunarLander-v2'
env = 'Acrobot-v1'
# env = 'MountainCar-v0'
# algo_to_config = {'logu': cartpole_hparams0, 'dqn': cartpole_dqn}
# env_to

def runner(algo):
    if algo == 'logu':
        if env == 'CartPole-v1':
            config = cartpole_hparams2
        elif env == 'MountainCar-v0':
            config = mcar_hparams
        elif env == 'Acrobot-v1':
            config = acrobot_logu2
        elif env == 'LunarLander-v2':
            config = lunar_hparams_logu
        algo = LogULearner
    elif algo == 'dqn':
        config = cartpole_dqn
        algo = CustomDQN
    elif algo == 'ppo':
        if env == 'CartPole-v1':
            config = cartpole_ppo
        elif env == 'Acrobot-v1':
            config = acrobot_ppo
        algo = CustomPPO

    model = algo(env, **config, log_dir='ft/acro',
                 device='cpu', log_interval=250, num_nets=1)
    model.learn(total_timesteps=50_000)


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
