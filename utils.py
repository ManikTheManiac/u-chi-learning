import os
from stable_baselines3.common.logger import configure
import time


def logger_at_folder(log_dir=None, algo_name=None):
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
