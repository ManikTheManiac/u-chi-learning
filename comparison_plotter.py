# from traceback import print_tb
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# from tbparse import SummaryReader

# sns.set_theme(style="darkgrid")
# algo_to_log_interval = {'DQN': 500, 'PPO': 4000,
#                         'LogU0': 500, 'RawLik': 500, 'LogU2nets': 500}

# desired_algos = ['PPO', 'newtuned', '1kls', 'acro1', 'min', 'min-theta', 'max-theta', 'max']

# def plotter(folder, metrics=['step', 'eval/avg_reward'], ylim=None):
#     # First, scan the folder for the different algorithms:
#     algos = []
#     plt.figure()
#     algo_data = pd.DataFrame()
#     subfolders = os.listdir(folder)
#     # Remove pngs
#     subfolders = [f for f in subfolders if not f.endswith('.png')]
#     for subfolder in subfolders:
#         algo_name = subfolder.split('_')[0]
#         if algo_name not in desired_algos:
#             continue

#         if algo_name not in algos:
#             algos.append(algo_name)
        
#         subfiles = os.listdir(f'{folder}/{subfolder}')
#         # ignore csvs:
#         file = subfiles[0]

#         # Convert the tensorboard file to a pandas dataframe:
#         log_file = f'{folder}/{subfolder}/{file}'
#         print("Processing", subfolder)

#         reader = SummaryReader(log_file)
#         df = reader.scalars
#         # filter the desired metrics:

#         try:
#             df = df[df['tag'].isin(metrics)]
#             # df.plot(x='step', y='value', label=algo_name)
#             # Add a column with this data:
#             df['algo'] = algo_name
#             # Add a column with the run number:
#             df['run'] = subfolder.split('_')[1]
#             # Add the df to the algo_data:
#             # algo_data = algo_data.append(df)
#             # convert this to a concat of dataframes:
#             algo_data = pd.concat([algo_data, df])
#         except Exception as e:
#             print("Error processing", log_file)
#             print_tb(e.__traceback__)
#             continue

#     sns.lineplot(data=algo_data, x='step', y='value', hue='algo')
#     # Append the number of runs to the legend for each algo:
#     for algo in algos:
#         plt.plot([], [], ' ', label=f'{algo} ({len(algo_data[algo_data["algo"] == algo]["run"].unique())} runs)')
#     plt.legend()
#     if ylim is not None:
#         plt.ylim(ylim)
#     plt.xlabel('Environment Steps')
#     plt.ylabel(metrics[1])
#     # Use the y value as the filename, but strip before the first slash:
#     try:
#         name = metrics[1].split('/')[1]
#     except:
#         name = metrics[1]

#     plt.savefig(f'{folder}/{name}.png')


# if __name__ == '__main__':
#     # plotter('ft/benchmark/cartpole')
#     # plotter('ft/benchmark/mountaincar')

#     folder = 'ft/acro'
#     # folder = 'ft/benchmark'
#     # folder = 'multinasium'
#     # plotter(folder=folder, metrics=['step', 'train/loss', 'loss'])
#     plotter(folder=folder,  metrics=['step', 'eval/avg_reward'])
#     plotter(folder=folder, metrics=['step', 'rollout/reward'])
#     plotter(folder=folder, metrics=['step', 'train/theta', 'theta'])
#     plotter(folder=folder, metrics=['step', 'train/avg logu', 'avg logu'])

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from glob import glob
from tbparse import SummaryReader

sns.set_theme(style="darkgrid")
desired_algos = ['PPO', 'newtuned', '1kls', 'acro1', 'min', 'min-theta', 'max-theta', 'max']

def plotter(folder, metrics=['step', 'eval/avg_reward'], ylim=None):
    plt.figure()
    algo_data = pd.DataFrame()
    subfolders = glob(os.path.join(folder, '*'))

    for subfolder in subfolders:
        if not os.path.isdir(subfolder) or subfolder.endswith('.png'):
            continue

        algo_name = os.path.basename(subfolder).split('_')[0]
        if algo_name not in desired_algos:
            continue

        log_files = glob(os.path.join(subfolder, '*.tfevents.*'))
        if not log_files:
            print(f"No log files found in {subfolder}")
            continue

        log_file = log_files[0]
        print("Processing", os.path.basename(subfolder))

        try:
            reader = SummaryReader(log_file)
            df = reader.scalars
            df = df[df['tag'].isin(metrics)]
            df['algo'] = algo_name
            df['run'] = os.path.basename(subfolder).split('_')[1]
            algo_data = pd.concat([algo_data, df])
        except Exception as e:
            print("Error processing", log_file)
            continue

    if not algo_data.empty:
        sns.lineplot(data=algo_data, x='step', y='value', hue='algo')

        # Append the number of runs to the legend for each algo:
        algo_runs = algo_data.groupby('algo')['run'].nunique()
        for algo, runs in algo_runs.items():
            plt.plot([], [], ' ', label=f'{algo} ({runs} runs)')
        plt.legend()

        if ylim is not None:
            plt.ylim(ylim)
        plt.xlabel('Environment Steps')
        plt.ylabel(metrics[1].split('/')[-1])

        plt.savefig(os.path.join(folder, f"{metrics[1].split('/')[-1]}.png"))
        plt.close()
    else:
        print("No data to plot.")

if __name__ == "__main__":
    folder = 'ft/acro'
    plotter(folder=folder, metrics=['step', 'eval/avg_reward'])
    plotter(folder=folder, metrics=['step', 'rollout/reward'])
    plotter(folder=folder, metrics=['step', 'train/theta', 'theta'])
    plotter(folder=folder, metrics=['step', 'train/avg logu', 'avg logu'])