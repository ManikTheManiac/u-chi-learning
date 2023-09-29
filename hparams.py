cartpole_hparams0 = {
    'beta': 0.1,
    'batch_size': 400,
    'buffer_size': 150000,
    'gradient_steps': 20,
    'learning_rate': 5.5e-3,
    'target_update_interval': 70,
    'tau': 0.85,
    'tau_theta': 0.85,
    'hidden_dim': 16,
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
    'batch_size': 64,
    'buffer_size': 100000,
    'exploration_final_eps': 0.04,
    'exploration_fraction': 0.12,
    'gamma': 0.99,
    'gradient_steps': 128,
    'hidden_dim': 256,
    'learning_rate': 0.0023,
    'learning_starts': 1000,
    'target_update_interval': 10,
    'tau': 1.0,
    'train_freq': 256,
}