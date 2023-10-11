LogU learning implementation in gym (mazes and cartpole)

# New (Simple) Features:
- [x] Monitor FPS
- [ ] Monitor min/max of logu to watch for divergence
- [ ] Add learning rate decay thru scheduler
- [x] Add "train_freq" rather than episodic trainin:
- [ ] Possibly use SB3 style: :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
- [x] Add gradient clipping
- [ ] More clever normalization to avoid logu divergence
- [ ] Merge Rawlik with LogU as an option. e.g. prior_update_interval=0 for no updates, and otherwise use Rawlik iteration
- [x] Switch to SB3 Replay Buffer

# Experimental questions:
- [ ] Does stabilizing theta help stabilize logu? (i.e. fix theta to g.t. value)
- [ ] Test the use of clipping theta (min_reward, max_reward) and logu (no theoretical bounds, but -50/50 after norm. to avoid divergence)
- [ ] Which params most strongly affect logu oscillations?
- [ ] "..." affect logu divergence? 
- [ ] Why does using off-policy (pi0) for exploration make logu diverge?
- [ ] Which activation function is best?
- [ ] Which aggregration of theta is best (min/mean/max), same for logu (min is suggested to help with over-optimistic behavior)

# Features requiring experiments:
- [ ] use target or online logu for exploration (greedy or not?)
- [ ] Standard prioritized replay
- [ ] Clipping theta
- [ ] smooth out theta learning

# Future TODOs:
- [ ] Generate dependencies
- [ ] Write tests
- [ ] Make more off-policy / offline?
- [ ] V learning with cloning
- [x] UV learning
- [ ] Rawlik scheme

# Notes:
- I had to change this in SB3 code to allow for next_actions in replay buffer (stable_baselines3.common.type_aliases)
class ReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    next_actions: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor

I added this line to the `gymnasium/envs/__init__.py` file:
```
register(
    id="Simple-v0",
    entry_point="gymnasium.envs.classic_control.simple_env:SimpleEnv",
    max_episode_steps=10,
    reward_threshold=1.0,
)
```

Model-based ground truth comparisons with tabular algorithms:

![eigvec](figures/left_eigenvector_MB.png)
![policy](figures/policy_MB.png)

Model-free ground truth comparisons:

![eigvec][eigvec_figure]
![policy][policy_figure]

[policy_figure]: figures/policy_MF.png
[eigvec_figure]: figures/left_eigenvector_MF.png