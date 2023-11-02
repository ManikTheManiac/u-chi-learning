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
# env = 'Acrobot-v1'
env = 'MountainCar-v0'

str_to_algo = {
    'logu': LogULearner,
    'ppo': CustomPPO,
    'dqn': CustomDQN
}



def runner(algo):
    if env == 'MountainCar-v0':
        configs = mcars
    elif env == 'CartPole-v1':
        configs = cartpoles
    # elif env == 'LunarLander-v2':
    #     configs = lunars
    elif env == 'Acrobot-v1':
        configs = acrobots

    # Now access the config for this algo:
    config = configs[algo]
    algo = str_to_algo[algo]

    model = algo(env, **config, log_dir='ft/mcar',
                 device='cuda', log_interval=250)
                #, num_nets=2, aggregator='max')
    model.learn(total_timesteps=120_000)


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
