# Plot the results from scaling_expt.py
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
sns.set_theme(style="darkgrid")

folder = 'scaling_expt_results'
env_id = 'Acrobot-v1'
algo = 'dqn'
df = pd.read_csv(os.path.join(folder, f'{env_id}-{algo}.txt'), header=None)
df.columns = ['hidden_dim', 'auc'] 
df = df.sort_values(by='hidden_dim')
# take average over same hidden_dim:
df = df.groupby('hidden_dim')
# Get mean and std:
df = df.agg(['mean', 'std'])
df = df.reset_index()
df.columns = ['hidden_dim', 'auc', 'auc_std']
# Plot:
plt.figure()
# mean:
# plt.plot(df['hidden_dim'], df['auc'], 'ko', label='mean')
# std as error bars (center the bars on the mean):
plt.errorbar(df['hidden_dim'], df['auc'], yerr=df['auc_std'], label='std', fmt='o')

plt.xlabel('Nodes in each hidden layer')
plt.ylabel('Area under eval reward curve')
plt.xscale('log')
plt.title(f'{env_id}')
# tight layout:
plt.tight_layout()
# Draw a star at 64 (where hparam was optimized), plot on top of line:
plt.plot(64, df[df['hidden_dim']==64]['auc'], marker='*', markersize=20, c='y')
plt.savefig(f'{env_id}-{algo}.png')
plt.close()