import argparse
import wandb
# from darer.MultiLogU import LogULearner
from LogUAC import LogUActor

# env_id = 'CartPole-v1'
# env_id = 'MountainCar-v0'
# env_id = 'HalfCheetah-v4'
env_id = 'Pendulum-v1'


def runner(config=None, run=None):
    # Convert nec kwargs to ints:
    for int_kwarg in ['batch_size', 'buffer_size', 'gradient_steps', 'target_update_interval', 'train_freq']:
        config[int_kwarg] = int(config[int_kwarg])
    # Remove the "learning_starts" kwarg, for now:
    # config.pop('learning_starts')
    # config.pop('policy_kwargs')
    runs_per_hparam = 2
    auc = 0
    wandb.log({'env_id': env_id})

    for _ in range(runs_per_hparam):
        model = LogUActor(env_id, **config, log_interval=1000, device='cpu')
        model.learn(total_timesteps=50_000)
        auc += model.eval_auc
    auc /= runs_per_hparam
    wandb.log({'avg_eval_auc': auc})


def wandb_agent():
    with wandb.init(sync_tensorboard=False, monitor_gym=False, dir='logs') as run:
        cfg = run.config
        dict_cfg = cfg.as_dict()
        # Add args.apply_clips to the config dict:
        # dict_cfg['hidden_dim'] = args.hidden_dim

        runner(dict_cfg, run=run)


if __name__ == "__main__":
    # Parse the "algo" argument
    parser = argparse.ArgumentParser()
    # parser.add_argument("-hd", "--hidden_dim", type=int, default=256)
    parser.add_argument("-c", "--count", type=int, default=100)
    parser.add_argument("--entity", type=str, default="jacobhadamczyk")
    parser.add_argument("--project", type=str, default="LogU-Cartpole")
    parser.add_argument("--sweep_id", type=str, default="ynz7zay1")
    args = parser.parse_args()
    full_sweep_id = f"{args.entity}/{args.project}/{args.sweep_id}"

    # Before calling the agent on this full_sweep_id, make sure it exists (i.e. the project and sweep):
    # test_sweep_existence(full_sweep_id)
    # def configured_agent():
    #     return wandb_agent()

    wandb.agent(full_sweep_id, function=wandb_agent, count=args.count)
