import argparse
import wandb
from darer.MultiLogU import LogULearner
from new_logac import LogUActor

# env_id = 'CartPole-v1'
# env_id = 'MountainCar-v0'
env_id = 'LunarLander-v2'
env_id = 'Pong-v4'
# env_id = 'HalfCheetah-v4'
# env_id = 'Pendulum-v1'


def runner(config=None, run=None, device='cpu'):
    # Convert the necessary kwargs to ints:
    for int_kwarg in ['batch_size', 'target_update_interval', 'theta_update_interval']:
        config[int_kwarg] = int(config[int_kwarg])
    config['buffer_size'] = 200_000
    config['gradient_steps'] = 50
    config['train_freq'] = 200
    config['learning_starts'] = 25_000

    config.pop('actor_learning_rate')
    runs_per_hparam = 2
    auc = 0
    wandb.log({'env_id': env_id})

    for _ in range(runs_per_hparam):
        model = LogULearner(env_id, **config, log_interval=1000, use_wandb=True,
                            device=device, render=0)
        model.learn(total_timesteps=200_000)
        auc += model.eval_auc
    auc /= runs_per_hparam
    wandb.log({'avg_eval_auc': auc})


def wandb_agent():
    with wandb.init(sync_tensorboard=False, monitor_gym=False, dir='logs') as run:
        cfg = run.config
        dict_cfg = cfg.as_dict()
        runner(dict_cfg, run=run, device=args.device)


if __name__ == "__main__":
    # set up wandb variables (TODO: these should be set up as globals per user):
    entity = "jacobhadamczyk"
    project = "LogU-Cartpole"
    sweep_id = "rbtzmhyx"
    # Parse the "algo" argument
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", type=str, default='cuda')
    parser.add_argument("-c", "--count", type=int, default=100)

    args = parser.parse_args()
    full_sweep_id = f"{entity}/{project}/{sweep_id}"

    # TODO: Before calling the agent on this full_sweep_id, make sure it exists (i.e. the project and sweep):
    # test_sweep_existence(full_sweep_id)
    wandb.agent(full_sweep_id, function=wandb_agent, count=args.count)
