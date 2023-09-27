import os
from stable_baselines3.common.logger import configure

def logger_at_folder(log_dir='', algo_name=''):
    # Generate a logger object at the specified folder:
    if log_dir != '':
        files = os.listdir(log_dir)
        # Get the number of existing "LogU" directories:
        num = len([int(f.split('_')[1]) for f in files if 'LogU' in f]) + 1
        tmp_path = f"{log_dir}/LogU_{num}"
    
        os.makedirs(tmp_path, exist_ok=True)
        logger = configure(tmp_path, ["stdout", "tensorboard"])
    else:
        # print the logs to stdout:
        logger = configure(format_strings=["stdout"])#, "csv", "tensorboard"])

    return logger