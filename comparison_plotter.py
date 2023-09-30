from traceback import print_tb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tbparse import SummaryReader

sns.set_theme(style="darkgrid")
algo_to_log_interval = {'DQN': 500, 'PPO': 4000,
                        'LogU0': 500, 'RawLik': 500, 'LogU2nets': 500}


def plotter(folder, metrics=['step', 'eval/avg_reward']):
    # First, scan the folder for the different algorithms:
    algos = []
    plt.figure()
    algo_data = pd.DataFrame()
    for subfolder in os.listdir(folder):
        algo_name = subfolder.split('_')[0]
        if algo_name not in algos:
            algos.append(algo_name)

        subfiles = os.listdir(f'{folder}/{subfolder}')
        # ignore csvs:
        subfiles = [f for f in subfiles if not f.endswith('.csv')]
        file = subfiles[0]

        # Convert the tensorboard file to a pandas dataframe:
        log_file = f'{folder}/{subfolder}/{file}'
        print("Processing", log_file, "...")
        reader = SummaryReader(log_file)
        df = reader.scalars
        # filter the desired metrics:

        try:
            df = df[df['tag'].isin(metrics)]
            # df.plot(x='step', y='value', label=algo_name)
            # Add a column with this data:
            df['algo'] = algo_name
            # Add a column with the run number:
            df['run'] = subfolder.split('_')[1]
            # Add the df to the algo_data:
            # algo_data = algo_data.append(df)
            # convert this to a concat of dataframes:
            algo_data = pd.concat([algo_data, df])
        except Exception as e:
            print("Error processing", log_file)
            print_tb(e.__traceback__)
            continue
    # Now, plot the algo_data:
    sns.lineplot(data=algo_data, x='step', y='value', hue='algo')
    plt.legend()
    # plt.xlim(0, 100000)
    plt.xlabel('Environment Steps')
    plt.ylabel('Average Evaluation Reward')
    plt.savefig(f'{folder}_rwds.png')
    # plt.show()


if __name__ == '__main__':
    plotter('ft/benchmark/cartpole')
    plotter('ft/benchmark/mountaincar')
