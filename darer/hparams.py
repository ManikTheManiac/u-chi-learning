cartpole_hparams0 = {
    'beta': 0.1,
    'batch_size': 300,
    'buffer_size': 150000,
    'gradient_steps': 1,
    'learning_rate': 2e-3,
    'target_update_interval': 70,
    'tau': 0.85,
    'theta_update_interval': 1,
    'tau_theta': 0.4,
    'hidden_dim': 64,
    'train_freq': 1,
    'learning_starts': 1000
}

cartpole_hparams2 = {
    'batch_size': 860,
    'beta': 10,
    'buffer_size': 100_000,
    'gradient_steps': 1,
    'learning_rate': 2.5e-2,
    'target_update_interval': 30,
    'tau': 0.34,
    'tau_theta': 0.967,
    'theta_update_interval': 8,
    'hidden_dim': 64,
    'train_freq': 1,
    'learning_starts': 1000
}
mcar_hparams = {
    'beta': 0.78,
    'batch_size': 950,
    'buffer_size': 53000,
    'gradient_steps': 24,
    'learning_rate': 7.2e-4,
    'target_update_interval': 270,
    'tau': 0.28,
    'tau_theta': 0.23,
    'hidden_dim': 64,
    'train_freq': 125,
    'learning_starts': 5000
}

lunar_hparams = {
    'beta': 0.18,
    'batch_size': 800,
    'buffer_size': 1_000_000,
    'gradient_steps': 1,
    'learning_rate': 7.2e-5,
    'target_update_interval': 270,
    'tau': 0.28,
    'tau_theta': 0.23,
    'hidden_dim': 64,
    'train_freq': 12
}

mcar_hparams2 = {
    'beta': 70,
    'batch_size': 256,
    'buffer_size': 1_300_000,
    'gradient_steps': 1,
    'learning_rate': 5e-4,
    'target_update_interval': 5000,
    'tau': 0.95,
    'tau_theta': 0.0001,
    'hidden_dim': 128,
    'train_freq': 1
}

sac_hparams2 = {
    'beta': 80,
    'batch_size': 32,
    'buffer_size': 1_000_000,
    'gradient_steps': 1,
    'learning_rate': 3e-4,
    'target_update_interval': 1,
    'tau': 0.005,
    'tau_theta': 0.995,
    'hidden_dim': 64,
    'train_freq': 1
}


easy_hparams2 = {
    'beta': 3,
    'batch_size': 512,
    'buffer_size': 1_000_000,
    'gradient_steps': 1,
    'learning_rate': 3e-4,
    'target_update_interval': 1,
    'tau': 0.005,
    'tau_theta': 0.998,
    'hidden_dim': 256,
    'train_freq': 1
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

acrobot_ppo = {
    'ent_coef': 0,
    'gae_lambda': 0.94,
    'n_epochs': 4,
    'n_steps': 256,
    # 'normalize': True,
    # 'normalize_kwargs': {'norm_obs': True, 'norm_reward': False}
}

cartpole_hparams1 = {
    'beta': 1.0,
    'batch_size': 256,
    'buffer_size': 100_000,
    'gradient_steps': 1,
    'learning_rate': 3.6e-3,
    'target_update_interval': 125,
    'tau': 0.7,
    'tau_theta': 0.7,
    'train_freq': 4,
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

acrobot_logu = {
    'beta': 0.2,
    'batch_size': 855,
    'buffer_size': 100_000,
    'gradient_steps': 1,
    'learning_rate': 7.e-3,
    'target_update_interval': 6,
    'tau': 0.57,
    'tau_theta': 0.90,
    'train_freq': 1,
    'hidden_dim': 64,
    'learning_starts': 1_000
}

cheetah_hparams = {
    'batch_size': 600,
    'beta': 10,
    'buffer_size': 1_000_000,
    'gradient_steps': 1,
    'learning_rate': 8e-6,
    'target_update_interval': 10,
    'tau': 0.99,
    'tau_theta': 0.9,
    'train_freq': 1,
    'hidden_dim': 64,
}


cheetah_hparams2 = {
    'batch_size': 200,
    'beta': 10,#0.87,
    'buffer_size': 1_000_000,
    'gradient_steps': 1,
    'learning_rate': 3.e-4,
    'actor_learning_rate': 1.e-5,
    'target_update_interval': 8,
    'theta_update_interval': 10,
    'tau': 0.84,
    'tau_theta': 0.955,
    'train_freq': 1,
    'hidden_dim': 256,
}

lunar_hparams_logu = {
    'batch_size': 200,
    'beta': 6.,
    'hidden_dim': 256,
    'learning_rate': 6e-4,
    'target_update_interval': 1500,
    'tau': 0.3,
    'tau_theta': 0.95,
    'theta_update_interval': 35,
    'buffer_size': 50_000,
    'gradient_steps': 4,
    'train_freq': 12,
}