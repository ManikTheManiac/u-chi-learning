import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tbparse import SummaryReader

sns.set_theme(style="darkgrid")

def plotter(folder, metrics=['eval/avg_reward']):
    # First, scan the folder for the different algorithms:
    algos = []
    algo_to_data = {}
    for subfolder in os.listdir(folder):
        algo_name = subfolder.split('_')[0]
        if algo_name not in algos:
            algos.append(algo_name)
            algo_to_data[algo_name] = pd.DataFrame()

    # iterate through the files in the folder
    for subfolder in os.listdir(folder):
        # Extract algorithm name, and add it to the df:
        algo_name = subfolder.split('_')[0]
        # There should be only one file in the subfolder:
        subfiles = os.listdir(f'{folder}/{subfolder}')
        # ignore csvs:
        subfiles = [f for f in subfiles if not f.endswith('.csv')]
        file = subfiles[0]
        
        # Convert the tensorboard file to a pandas dataframe:
        log_file = f'{folder}/{subfolder}/{file}'
        reader = SummaryReader(log_file)
        df = reader.scalars
        # filter the desired metrics:
        # strip the : from the metrics:
        df = df[df['tag'].isin(metrics)]
        # Add the data from the file to the new df by cat, using algo_name+num as the column name:
        algo_df = algo_to_data[algo_name]
        algo_df = pd.concat([algo_df, df], axis=1)
        algo_to_data[algo_name] = algo_df

    plt.figure()
    # now, plot the dfs from each algo:
    for algo_name in algos:
        data = algo_to_data[algo_name]
        # plot with errors:
        log_interval = 500
        t_axis = np.arange(0, len(data) * log_interval, log_interval)
        # take an average of the data_df:
        means = data['value'].mean(axis=1)
        num_runs = data['value'].shape[1]
        stds = data['value'].std(axis=1) / np.sqrt(num_runs)
        # plot with errors:
        plt.plot(t_axis, means, label=algo_name)
        plt.fill_between(t_axis, means - stds, means + stds, alpha=0.5)

    plt.legend()
    plt.xlabel('Environment Steps')
    plt.ylabel('Average Evaluation Reward')
    plt.savefig(f'{folder}_rwds.png')
    plt.show()

if __name__ == '__main__':
    plotter('ft/benchmark')
