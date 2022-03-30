import gym
import numpy as np
from rlutils import make, train
from rlutils.observers.observer import Observer


train_parameters = {
    "project_name":         "pendulum",
    "env":                  "Pendulum-v0",
    "state_dims":           ["cos_theta","sin_theta","theta_dot"],

    # "project_name":         "mountaincar",
    # "env":                  "MountainCarContinuous-v0",
    # "state_dims":           ["pos","vel"],

    "agent":                "mbpo",
    "num_episodes":         100,
    "episode_time_limit":   200,
    "from_pixels":          False,
    "wandb_monitor":        False,
    "render_freq":          0,
    "video_save_freq":      0,
    "video_to_wandb":       True,
    "observe_freq":         0,
    "checkpoint_freq":      0,
}

if train_parameters["env"] == "Pendulum-v0":
    from rlutils.specific.Pendulum import reward_function

# Make environment.
env = gym.make(train_parameters["env"]).unwrapped # Needed to impose custom time limit

agent_parameters = {}
agent_parameters["ddpg"] = {
    "input_normaliser":     "box_bounds",
    "replay_capacity":      5000,
    "batch_size":           32,
    "lr_pi":                1e-4,
    "lr_Q":                 1e-3,
    "gamma":                0.99,
    "tau":                  0.01,
    "noise_params":         ("ou", 0., 0.15, 0.5, 0.01, 100),
}
agent_parameters["td3"] = {**agent_parameters["ddpg"], **{ 
    "td3":                  True,
    "td3_noise_std":        0.2,
    "td3_noise_clip":       0.5,
    "td3_policy_freq":      2
}}    
agent_parameters["sac"] = {
    "input_normaliser":     "box_bounds",
    "replay_capacity":      5000,
    "batch_size":           32,
    "lr_pi":                1e-4,
    "lr_Q":                 1e-3,
    "gamma":                0.99,
    "alpha":                0.2,
    "tau":                  0.01,
}
agent_parameters["ppo"] = {
    "lr_pi": 3e-4,       
    "lr_V": 1e-3,
    "num_steps_per_update": 80, # Number of gradient steps per update.
    "baseline": "Z", # Baselining method: either "off", "Z" or "adv".
    "epsilon": 0.2, # Clip ratio for policy update.
    "noise_params": ("norm", 0.6, 0.1, 0.05, 2000), # Initial std, final std, decay rate, decay freq (timesteps).
}
agent_parameters["diayn"] = {
    "input_normaliser":     "box_bounds",
    "num_skills":           20, 
    "batch_size":           128,
    "alpha":                0.1,
    "tau":                  0.01
}
agent_parameters["simple_model_based"] = {
    "input_normaliser":     "box_bounds",
    "reward":               reward_function,
    "ensemble_size":        5,
    "probabilistic":        False,
    "model_freq":           1,
    "batch_size":           32, 
    "num_random_steps":     0,
    "batch_ratio":          1,
}
agent_parameters["steve"] = {**agent_parameters["td3"], **{
    "input_normaliser":     "box_bounds",
    "reward":               reward_function,
    "ensemble_size":        2,
    "num_random_steps":     0,
    "batch_size":           32,
    "rollout": {
        "horizon_params":   ("constant", 5)
    }
}}
agent_parameters["mbpo"] = {**agent_parameters["sac"], **{
    "input_normaliser":     "box_bounds",
    "reward":               reward_function,
    "probabilistic":        False,
    "num_random_steps":     0,
    "model_freq":           1,
    "rollouts_per_update":  2,
    "retained_updates":     500,
    "policy_updates_per_timestep": 20,
    "rollout": {
      "horizon_params": ("linear", 1, 25, (1000, 3000))
    }
}}

a = train_parameters["agent"]
agent = make(a, env, agent_parameters[a])
print(agent)
obs = Observer(P={"save_freq": np.inf}, state_dims=train_parameters["state_dims"], action_dims=1)
_, rn = train(agent, train_parameters, observers={"observer": obs})

if train_parameters["observe_freq"]:
    obs.add_custom_dims(lambda x: np.array([np.arccos(x[2]) * np.sign(x[3])]), ["theta"])
    obs.add_future(["reward"], gamma=agent.P["gamma"], new_dims=["return"]) # Add return dim.
    obs.save(f"runs/{rn}_train.csv")
