# We run a scaling experiment on acrobot to show how hidden_dim affects performance:
from MultiLogU import LogULearner
from hparams import *
hparams = acrobot_logu

env_id = 'Acrobot-v1'

def runner(hidden_dim, device='cpu', total_timesteps=20_000):
    hparams.pop('hidden_dim')

    hparams['hidden_dim'] = hidden_dim
    agent = LogULearner(env_id, **hparams, log_interval=1000, use_wandb=False,
                device=device, render=0)
    agent.learn(total_timesteps=total_timesteps)
    return agent.eval_auc


HIDDEN_DIMS = [2,3,4,5,6,7,8,9,10,12,16,32,64,96,128]
def main(chunk, chunk_size=3):
    # Grab a set of hidden_dims for a cpu to run:
    idxs = list(range(chunk*chunk_size, (chunk+1)*chunk_size))
    print(idxs)
    hidden_dims = HIDDEN_DIMS[idxs[0]:idxs[-1]+1]
    for hidden_dim in hidden_dims:
        print(f"Training with hidden_dim: {hidden_dim}")
        auc = runner(hidden_dim, device='cpu', total_timesteps=20_000)
        print(f"auc: {auc}")
        # Add new line to file:

        with open(f"scaling_expt_results/{env_id}.txt", 'a+') as f:
            f.write(f"{hidden_dim},{auc}\n")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--chunk', type=int, default=0)
    chunk = parser.parse_args().chunk
    main(chunk=chunk)