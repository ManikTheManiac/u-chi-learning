cartpole_hparams0 = {
    'beta': 0.1,
    'batch_size': 400,
    'buffer_size': 150000,
    'gradient_steps': 20,
    'learning_rate': 5.5e-3,
    'target_update_interval': 70,
    'tau': 0.85,
    'tau_theta': 0.85,
}

cartpole_hparams1 = {
    'beta': 2.0,
    'batch_size': 630,
    'buffer_size': 14000,
    'gradient_steps': 11,
    'learning_rate': 3.6e-3,
    'target_update_interval': 125,
    'tau': 0.82,
    'tau_theta': 0.52,
    'train_freq': 17,
    'hidden_dim': 512,
}

cartpole_dqn = {
    'batch_size': 560,
    'buffer_size': 15000,
    'exploration_final_eps': 0.037,
    'exploration_fraction': 0.31,
    'gamma': 0.975,
    'gradient_steps': 8,
    'hidden_dim': 512,
    'learning_rate': 0.00245,
    'learning_starts': 1500,
    'target_update_interval': 700,
    'tau': 0.465,
    'train_freq': 49,
}