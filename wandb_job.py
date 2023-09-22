import argparse
import wandb
import gym
from LogU import LogULearner
from gym.wrappers import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecVideoRecorder

# env = gym.make("MountainCar-v0")
env = gym.make('CartPole-v1')
env_id = 'CartPole-v1'
# env = gym.make('FrozenLake-v1')
# env = VecMonitor(env)
# env = Monitor(env, './logs', force=True)

# env = VecVideoRecorder(env, f"videos/", record_video_trigger=lambda x: x % 2000 == 0, video_length=200)


def runner(config=None, run=None):
    # Convert nec kwargs to ints:
    for int_kwarg in ['batch_size', 'buffer_size', 'gradient_steps', 'target_update_interval']:
        config[int_kwarg] = int(config[int_kwarg])
    # Remove the "learning_starts" kwarg, for now:
    # config.pop('learning_starts')
    # config.pop('policy_kwargs')

    model = LogULearner(env_id, **config, log_interval=500)

    model.learn_online(total_timesteps=150_000)


def wandb_agent():
    with wandb.init(sync_tensorboard=True,monitor_gym=True, dir='logs') as run:
        cfg = run.config
        dict_cfg = cfg.as_dict()
        # Add args.apply_clips to the config dict:
        dict_cfg['hidden_dim'] = args.hidden_dim

        runner(dict_cfg, run=run)

if __name__ == "__main__":
    entity = "jacobhadamczyk"
    project = "LogU-Cartpole"
    sweep_id = "cmoskhlk"
    # Parse the "algo" argument
    parser = argparse.ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, default=256)
    parser.add_argument("-c", "--count", type=int, default=100)
    
    args = parser.parse_args()

    full_sweep_id = f"{entity}/{project}/{sweep_id}"

    # Before calling the agent on this full_sweep_id, make sure it exists (i.e. the project and sweep):
    # test_sweep_existence(full_sweep_id)
    # def configured_agent():
    #     return wandb_agent()

    wandb.agent(full_sweep_id, function=wandb_agent, count=args.count)