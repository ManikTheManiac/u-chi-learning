LogU learning implementation in gym (mazes and cartpole)

# New Features:
- [ ] Monitor FPS

# Experimental questions:
- [ ] Does stabilizing theta help stabilize logu? (i.e. fix theta to g.t.)

# Features requiring experiments:
- [ ] use target or online logu for exploration (greedy or not?)
- [ ] Standard prioritized replay
- [ ] Clipping theta
- [ ] smooth out theta learning

# TODOs:
- [ ] Generate dependencies
- [ ] Write tests
- [ ] Make more off-policy / offline?
- [ ] V learning with cloning
- [ ] UV learning
- [ ] Rawlik scheme


Model-based ground truth comparisons with tabular algorithms:

![eigvec](figures/left_eigenvector_MB.png)
![policy](figures/policy_MB.png)

Model-free ground truth comparisons:

![eigvec][eigvec_figure]
![policy][policy_figure]

[policy_figure]: figures/policy_MF.png
[eigvec_figure]: figures/left_eigenvector_MF.png