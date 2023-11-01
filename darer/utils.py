import os
import gymnasium as gym
from stable_baselines3.common.logger import configure
import time


def logger_at_folder(log_dir=None, algo_name=None):
    # ensure no _ in algo_name:
    if '_' in algo_name:
        print("WARNING: '_' not allowed in algo_name (used for indexing). Replacing with '-'.")
    algo_name = algo_name.replace('_', '-')
    # Generate a logger object at the specified folder:
    if log_dir is not None:
        files = os.listdir(log_dir)
        # Get the number of existing "LogU" directories:
        num = len([int(f.split('_')[1]) for f in files if algo_name in f]) + 1
        tmp_path = f"{log_dir}/{algo_name}_{num}"

        # If the path exists already, increment the number:
        while os.path.exists(tmp_path):
            num += 1
            tmp_path = f"{log_dir}/{algo_name}_{num}"
            time.sleep(0.2)
            # try:
            #     os.makedirs(tmp_path, exist_ok=False)
            # except FileExistsError:
            #     # try again with an incremented number:
            # pass
        logger = configure(tmp_path, ["stdout", "tensorboard"])
    else:
        # print the logs to stdout:
        # , "csv", "tensorboard"])
        logger = configure(format_strings=["stdout"])

    return logger

def env_id_to_envs(env_id, render):
    if isinstance(env_id, str):
        env = gym.make(env_id)
        # make another instance for evaluation purposes only:
        eval_env = gym.make(env_id, render_mode='human' if render else None)
    elif isinstance(env_id, gym.Env):
        env = env_id
        # Make a new copy for the eval env:
        import copy
        eval_env = copy.deepcopy(env_id)
    else:
        raise ValueError(
            "env_id must be a string or gym.Env instance.")

    return env, eval_env