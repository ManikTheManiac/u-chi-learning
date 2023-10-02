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
    'train_freq': 50,
    'log_interval': 1000
}

mcar_hparams = {
    'beta': 0.5,
    'batch_size': 400,
    'buffer_size': 150000,
    'gradient_steps': 20,
    'learning_rate': 5.5e-3,
    'target_update_interval': 70,
    'tau': 0.85,
    'tau_theta': 0.85,
    'hidden_dim': 128,
    'train_freq': 50
}

cartpole_rawlik = {
    'beta': 0.1,
    'batch_size': 400,
    'buffer_size': 150000,
    'gradient_steps': 20,
    'learning_rate': 1.5e-3,
    'target_update_interval': 170,
    'tau': 0.85,
    'tau_theta': 0.85,
    'hidden_dim': 64,
    'prior_update_interval': 2000
}

cartpole_ppo = {
    'batch_size': 256,
    'clip_range': 0.2,
    'ent_coef': 0.0,
    'gae_lambda': 0.8,
    'gamma': 0.98,
    'learning_rate': 0.001,
    'n_epochs': 20,
    'n_steps': 32,
    'hidden_dim': 64,
}

cartpole_hparams1 = {
    'beta': 1.0,
    'batch_size': 256,
    'buffer_size': 10_000,
    'gradient_steps': 4,
    'learning_rate': 3.6e-3,
    'target_update_interval': 125,
    'tau': 0.7,
    'tau_theta': 0.7,
    'train_freq': 1,
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
