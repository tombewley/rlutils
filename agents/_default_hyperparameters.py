"""
NOTE: Agent names must be lowercase.
"""

default_hyperparameters = {
  
  "actor_critic": {
    "net_pi": [(None, 64), "R", (64, 128), "R", (128, None), "S"], # Softmax policy.
    "net_V": [(None, 64), "R", (64, 128), "R", (128, None)],
    "input_normaliser": None, # Set to "box_bounds" to pre-normalise network inputs.
    "lr_pi": 1e-4, # Learning rate for policy.
    "lr_V": 1e-3, # Learning rate for state value function.
    "gamma": 0.99 # Discount factor.
  },   

  "diayn": { # See https://github.com/haarnoja/sac/blob/8258e33633c7e37833cc39315891e77adfbe14b2/examples/mujoco_all_diayn.py.
    "net_disc": [(None, 256), "R", (256, 256), "R", (256, None)], # Outputs log skill probabilities.
    "lr_disc": 3e-4, # Learning rate for discriminator.
    "num_skills": 50, # Number of skills. NOTE: Highly environment-dependent!
    "include_actions": False, # Whether or not to include action dimensions in discriminator input.
    "include_next_states": False, # Whether or not to include next state dimensions in discriminator input.
    "log_p_z_in_reward": True, # Whether or not to include -log(p(z)) term in pseudo-reward.
    "recompute_pseudo_reward_on_sample": True # Whether or not to recompute latest pseudo-rewards for a sampled batch.
  },

  "ddpg": {
    "net_pi": [(None, 256), "R", (256, 256), "R", (256, None), "T"], # Tanh policy (bounded in [-1,1]).
    "net_Q": [(None, 256), "R", (256, 256), "R", (256, 1)],
    "input_normaliser": None, # Set to "box_bounds" to pre-normalise network inputs.
    "replay_capacity": 50000, # Size of replay memory (starts overwriting when full).
    "batch_size": 128, # Size of batches to sample from replay memory during learning.
    "lr_pi": 1e-4, # Learning rate for policy.
    "lr_Q": 1e-3, # Learning rate for state-action value function.
    "gamma": 0.99, # Discount factor.
    "tau": 0.005, # Parameter for Polyak averaging of target network parameters.
    "noise_params": ("ou", 0., 0.15, 0.3, 0.3, 1000), # mu, theta, sigma_start, sigma_end, decay period (episodes).
    # "noise_params": ("un", 1, 0, 1000), # sigma_start, sigma_end, decay_period (episodes).
    "td3": False, # Whether or not to enable the TD3 enhancements. 
    # --- If TD3 enabled ---
    "td3_noise_std": 0.2,
    "td3_noise_clip": 0.5,
    "td3_policy_freq": 2
  },

  "dqn": {
    "net_Q": [(None, 256), "R", (256, 128), "R", (128, 64), "R", (64, None)], # From https://github.com/transedward/pytorch-dqn/blob/master/dqn_model.py.
    "input_normaliser": None, # Set to "box_bounds" to pre-normalise network inputs.
    "replay_capacity": 10000, # Size of replay memory (starts overwriting when full).
    "batch_size": 128, # Size of batches to sample from replay memory during learning.
    "lr_Q": 1e-3, # Learning rate for state-action value function.
    "gamma": 0.99, # Discount factor.
    "epsilon_start": 0.9,
    "epsilon_end": 0.05,
    "epsilon_decay": 500000, # Decay period (timesteps).
    "target_update": ("soft", 0.005), # Either ("hard", decay_period) or ("soft", tau).
    # "target_update": ("hard", 10000),
    "double": True, # Whether to enable double DQN variant to reduce overestimation bias.
    "reward_components": None # For reward decomposition (set to None to disable).
  },

  "forward_backward": {
    "net_FB": [(None, 256), "R", (256, 256), "R", (256, 256), "R", (256, None)],
    "input_normaliser": None,  # Set to "box_bounds" to pre-normalise network inputs.
    "embed_dim": 100,  # Dimensionality of embedding space.
    "cauchy_z": True,   # Whether to rescale sampled preference vectors using a Cauchy variable.
    "replay_capacity": int(1e6),  # Size of replay memory (starts overwriting when full).
    "batch_size": 128,  # Size of batches to sample from replay memory during learning.
    "lr_FB": 5e-4,  # Learning rate for F and B networks.
    "gamma": 0.99,  # Discount factor.
    "epsilon": 0.2,  # For behaviour policies.
    "softmax_tau": 0.2,  # For softmax policies used in updates.
    "tau": 0.005,  # Parameter for Polyak averaging of target network parameters.
    "lambda": 1.,  # Coefficient for orthonormality regularisation loss.
  },

  "mbpo": {
    "net_model": [(None, 200), "R", (200, 200), "R", (200, 200), "R", (200, 200), "R", (200, None)],
    "ensemble_size": 7, # Number of parallel dynamics models to train.
    "probabilistic": True, # Whether or not dynamics models output standard deviations alongside means.
    "num_random_steps": 1000, # Number of steps before disabling random mode and starting policy optimisation.
    "model_freq": 250, # Number of steps between model updates.
    "batch_size_model": 256, # Size of batch to use for model updates.
    "lr_model": 1e-3, # Learning rate for dynamics model.
    "rollouts_per_update": 400, # Number of rollouts to perform each time the model is updated.
    "rollout": {
      "horizon_params": ("linear", 1, 25, (20, 100)), # initial, final, (start of change, end of change) in units of model updates.
    },
    "retained_updates": 20, # Number of updates' worth of rollouts to retain in the simulated memory.
    "policy_updates_per_timestep": 20, # For pi/Q; use of model-generated data reduces overfitting risk.
  },

  "off_policy_mc": {
    "gamma": 0.99, # Discount factor.
    "epsilon": 0.5
  },

  "pets": {
    "net_model": [(None, 32), "R", (32, 64), "R", (64, None)],
    "input_normaliser": "box_bounds", # NOTE: especially beneficial for dynamics models.
    "ensemble_size": 5, # Number of dynamics models.
    "probabilistic": True, # Whether or not dynamics models output standard deviations alongside means.
    "delta_dynamics": True, # Whether or not dynamics models output *change* in state (c.f. next state itself).
    "num_random_steps": 2000, # Size of random replay memory (disables random mode when full).
    "batch_size": 256,
    "model_freq": 10, # Number of steps between model updates.
    "lr_model": 1e-3, # Learning rate for dynamics model.
    "replay_capacity": 2000, # Size of replay memory (starts overwriting when full).
    "batch_ratio": 0.9, # Proportion of on-policy transitions.
    "cem_iterations": 5, # Number of rounds of distribution refinement during planning.
    "cem_particles": 50,
    "cem_elites": 10,
    "cem_alpha": 0.1, # Update rate for CEM sampling distribution.
    "cem_temperature": 0.5, # Sharpness of elite weighting for MPPI extension.
    "cem_initial_inertia": 0., # Action inertia used to generate first iteration of particles.
    "cem_warm_start": False, # Whether to warm-start action means by time-shifting previous ones.
    "gamma": 0.99, # Discount factor.
    "rollout": {
      "horizon_params": ("constant", 20),
    }
  },

  "ppo": {
    "net_pi_cont": [(None, 64), "R", (64, 64), "R", (64, None), "T"], # Tanh policy (bounded in [-1,1]).
    "net_pi_disc": [(None, 64), "R", (64, 64), "R", (64, None), "S"], # Softmax policy for discrete.
    "net_V": [(None, 64), "R", (64, 64), "R", (64, None)],
    "input_normaliser": None, # Set to "box_bounds" to pre-normalise network inputs.
    "lr_pi": 3e-4,       
    "lr_V": 1e-3,
    "num_steps_per_update": 80, # Number of gradient steps per update.
    "gamma": 0.99, # Discount factor.
    "baseline": "Z", # Baselining method: either "off", "Z" or "adv".
    "epsilon": 0.2, # Clip ratio for policy update.
    "noise_params": ("norm", 0.6, 0.1, 0.05, int(2.5e5)), # Initial std, final std, decay rate, decay freq (timesteps).
  },

  "random": {
    "method": "uniform",
    "inertia": 0,
    "gamma": 0.99 # Discount factor.
  },

  "reinforce": {
    "net_pi": [(None, 64), "R", (64, 128), "R", (128, None), "S"], # Softmax policy.
    "input_normaliser": None, # Set to "box_bounds" to pre-normalise network inputs.
    "lr_pi": 1e-4, # Learning rate for policy.
    "gamma": 0.99, # Discount factor.
    "baseline": "adv", # Baselining method: either "off", "Z" or "adv".
    # --- If baseline == "adv" ---
    "net_V": [(None, 64), "R", (64, 128), "R", (128, None)],
    "lr_V": 1e-3, # Learning rate for state value function.
  },

  "sac": {
    "net_pi": [(None, 256), "R", (256, 256), "R", (256, None)],
    "net_Q": [(None, 256), "R", (256, 256), "R", (256, None)],
    "input_normaliser": None, # Set to "box_bounds" to pre-normalise network inputs.
    "replay_capacity": 10000, # Size of replay memory (starts overwriting when full).
    "batch_size": 256, # Size of batches to sample from replay memory during learning.
    "lr_pi": 1e-4, # Learning rate for policy.
    "lr_Q": 1e-3, # Learning rate for state-action value function.
    "gamma": 0.99, # Discount factor.
    "init_alpha": 0.1, # Initial weighting for entropy regularisation term.
    "learnable_alpha": True, # Whether to automatically tune alpha.
    "lr_alpha": 1e-4, # Learning rate for automatic alpha tuning.
    "tau": 0.005, # Parameter for Polyak averaging of target network parameters.
    "update_freq": 1, # Number of timesteps between updates.
  },
  
  "stable_baselines": { # NOTE: Other defaults specified in StableBaselines library.
    "model_class": "DQN",
    "verbose": True
  },

  "steve": {
    "net_model": [(None, 32), "R", (32, 64), "R", (64, None)],
    "ensemble_size": 2, # Number of parallel dynamics models to train.
    "num_random_steps": 1000, # Number of steps before disabling random mode and starting policy optimisation.
    "model_freq": 1, # Number of steps between model updates.
    "lr_model": 1e-3, # Learning rate for dynamics model.
    "rollout": {
      "horizon_params": ("constant", 5), # Maximum number of model steps to run to produce Q values.
    },
    "td3": True # STEVE is built around DDPG, and needs multiple Q_target networks.
  },

  "treeqn": {
    "net_node": [(None, 32), "R", (32, None)],
    "input_normaliser": None, # Set to "box_bounds" to pre-normalise network inputs.
    "replay_capacity": 10000, # Size of replay memory (starts overwriting when full).
    "batch_size": 128, # Size of batches to sample from replay memory during learning.
    "lr_Q": 1e-3, # Learning rate for state-action value function.
    "gamma": 0.99, # Discount factor.
    "epsilon_start": 0.9,
    "epsilon_end": 0.05,
    "epsilon_decay": 500000,
    "target_update": ("soft", 0.0005), # Either ("hard", decay_period) or ("soft", tau).
    # "target_update": ("hard", 10000),
    "double": True, # Whether to enable double DQN variant to reduce overestimation bias.
    "reward_components": None # For reward decomposition (set to None to disable).
  },

}