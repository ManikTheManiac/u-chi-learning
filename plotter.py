# Plot the results stored in the ft (finetuned) folder CSVs:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tbparse import SummaryReader

sns.set_theme(style="darkgrid")


import traceback
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

#
def plotter(folder):
    # create a new df with the same number of rows as the number of dfs in the folder
    data_df = pd.DataFrame()
    # iterate through the files in the folder
    for file_num, file in enumerate(os.listdir(folder)):
        # Convert the tensorboard file to a pandas dataframe:
        log_file = f'{folder}/{file}'
        # make sure it does not end with csv, png:
        if log_file.endswith('.csv') or log_file.endswith('.png'):
            continue
        reader = SummaryReader(log_file)
        df = reader.scalars
        # filter for the eval reward tag:
        try:
            # ensure length is 500:
            # if len(df) != 500:
            #     print(f'file {file} has length {len(df)}')
            #     continue
            df = df[df['tag'] == 'Eval. reward:']
            print(len(df))
            if len(df) == 60:
               data_df = pd.concat([data_df, df], axis=1)

        except:
            print(f'file {file} has no eval reward tag')
            continue
        # print(df)
        # Add the data from the file to the new df by cat:
    # now, plot the new df:
    t_axis = np.arange(0, 60, 1) * 500
    # take an average of the data_df:
    means = data_df['value'].mean(axis=1)
    num_runs = data_df['value'].shape[1]
    print(f'num_runs: {num_runs}')
    stds = data_df['value'].std(axis=1) / np.sqrt(num_runs)
    # calculate the 99% confidence interval bootstrap:
    # stds = data_df['value'].quantile(0.95, axis=1) - data_df['value'].quantile(0.05, axis=1)

    
    # plot with errors:
    plt.plot(t_axis, means)
    # plt.xlim(0,30000)
    plt.fill_between(t_axis, means - stds, means + stds, alpha=0.5)
    plt.xlabel('Timesteps')
    plt.ylabel('Reward')
    plt.title('Eval Reward vs Timesteps')
    plt.savefig(f'{folder}/eval_plot.png')
    plt.close()
    # Plot the individual runs, with smoothing over a window of w=50:
    # w = 10
    # plt.figure()
    # for i in range(num_runs):
    #     plt.plot(t_axis, data_df['value'].iloc[:,i].rolling(window=w).mean(), alpha=0.5)
    # plt.xlabel('Timesteps')
    # plt.ylabel('Reward')
    # plt.title('Eval Reward vs Timesteps')
    # plt.savefig(f'{folder}/eval_plot_runs.png')


if __name__ == "__main__":
    plotter('ft/cartpole6_0')
    # plotter('ft/pendulum')
    # plotter('ft/mountaincar_0')
    # plotter('ft/acrobot')