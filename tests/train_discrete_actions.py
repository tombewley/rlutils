import gym
from rlutils import make, train
from rlutils.observers.observer import Observer

"""
TODO:
- Ppo
- Treeqn
"""

train_parameters = {
    "project_name": "cartpole",
    "env":          "CartPole-v1",
    "state_dims":   ["pos","vel","ang","ang_vel"],

    "agent":                "actor_critic",

    "num_episodes":         1500,
    "episode_time_limit":   500,
    "from_pixels":          False,
    "wandb_monitor":        True,
    "render_freq":          0,
    "video_save_freq":      0,
    "video_to_wandb":       True,
    "observe_freq":         0,
    "checkpoint_freq":      0,
}

# Make environment.
env = gym.make(train_parameters["env"]).unwrapped # Needed to impose custom time limit
if train_parameters["from_pixels"]:
    # If from_pixels, set up screen processor.
    from rlutils.common.rendering import Renderer
    from rlutils.specific.CartPole import screen_processor # <<< NOTE: HARD CODED FOR CARTPOLE!
    env.reset()
    renderer = Renderer(env, screen_processor, mode="diff")
    renderer.get(first=True); env.step(0); s = renderer.get(show=False)
    state_shape = s.shape
else: state_shape, renderer = env.observation_space.shape, None

agent_parameters = {
    "dqn": {
        # "input_normaliser": "box_bounds",
        "replay_capacity":  10000,
        "batch_size":       32,
        "epsilon_start":    1,
        "epsilon_end":      0.05,
        "epsilon_decay":    10000,
        "target_update":    ("soft", 0.0005),
        "double":           True
    },
    "reinforce": {
        # "input_normaliser": "box_bounds",
        "lr_pi":            1e-4,
        "lr_V":             1e-3,
        "baseline":         "adv"
    },
     "actor_critic": {
        # "input_normaliser": "box_bounds",
        "lr_pi":            1e-4,
        "lr_V":             1e-3,
    }
}

a = train_parameters["agent"]
agent = make(a, env, agent_parameters[a])
print(agent)
obs = Observer(P={"save_freq": float("inf")}, state_dims=train_parameters["state_dims"], action_dims=1)
_, rn = train(agent, train_parameters, observers={"observer": obs})

if train_parameters["observe_freq"]:
    obs.add_future(["reward"], gamma=agent.P["gamma"], new_dims=["return"]) # Add return dim.
    obs.save(f"runs/{rn}_train.csv")
