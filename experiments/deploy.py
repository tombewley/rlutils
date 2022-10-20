from .. import from_numpy
from ..observers.loggers import SumLogger
from ..agents.stable_baselines import StableBaselinesAgent

from tqdm import tqdm
from gym import wrappers


P_DEFAULT = {"num_episodes": int(1e6), "render_freq": 1}


def train(agent, P=P_DEFAULT, wandb_config=None, save_dir="agents"):
    """
    Shortcut for training; just calls deploy with train=True.
    """
    return deploy(agent, P, True, wandb_config, save_dir)

def deploy(agent, P=P_DEFAULT, train=False, wandb_config=None, save_dir="agents"):

    do_extra = "do_extra" in P and P["do_extra"] # Whether or not to request extra predictions from the agent.
    do_wandb = wandb_config is not None
    do_render = "render_freq" in P and P["render_freq"] > 0
    do_checkpoints = "checkpoint_freq" in P and P["checkpoint_freq"] > 0
    if "observers" not in P: P["observers"] = {}

    if do_wandb: 
        # Initialise Weights & Biases monitoring.
        assert not type(agent)==StableBaselinesAgent, "wandb monitoring not implemented for StableBaselinesAgent."
        import wandb
        if "id" not in wandb_config:
            wandb_config["id"] = run_id = wandb.util.generate_id()
            wandb_config["resume"] = "never"
        else: wandb_config["resume"] = "must"
        run = wandb.init(**wandb_config,
            config={**agent.P, **P, **{n: o.P for n, o in P["observers"].items()}})
        run_name = run.name
    else:
        import time; run_id, run_name = None, time.strftime("%Y-%m-%d_%H-%M-%S")

    # Add observer for logging per-episode returns.
    P["observers"]["return_logger"] = SumLogger({"name": "return"})

    # Create directory for saving and tell observers what the run name is.
    if do_checkpoints: import os; save_dir += f"/{run_name}"; os.makedirs(save_dir, exist_ok=True) 
    for o in P["observers"].values(): o.run_names.append(run_name)

    # Add temporary wrappers to environment. NOTE: These are discarded when deployment is complete.
    env_inside_wrappers = agent.env
    if "episode_time_limit" in P: # Time limit.
        agent.env = wrappers.TimeLimit(agent.env, P["episode_time_limit"])
    if "video_freq" in P and P["video_freq"] > 0: # Video recording. NOTE: This must be the outermost wrapper.
        agent.env = wrappers.Monitor(agent.env, f"./video/{run_name}", video_callable=lambda ep: ep % P["video_freq"] == 0, force=True)

    # Stable Baselines uses its own training and saving procedures.
    if train and type(agent)==StableBaselinesAgent: agent.train(P["sb_parameters"])
    else:
        # Determine whether interaction budget is specified in units of episodes or steps.
        if "num_episodes" in P: assert "num_steps" not in P; by_steps = False
        else:                   assert "num_steps" in P;     by_steps = True; pbar = tqdm(total=int(P["num_steps"]), unit="step")
        force_break = False

        # Iterate through episodes.
        state = agent.env.reset()
        for ep in tqdm(range(P["num_episodes"]) if not by_steps else iter(int, 1), disable=by_steps, unit="episode"):
            render_this_ep = do_render and (ep+1) % P["render_freq"] == 0
            if render_this_ep: agent.env.render()
            
            # Get state in PyTorch format expected by agent.
            state_torch = from_numpy(state, device=agent.device)
            
            # Iterate through timesteps.
            t = 0; done = False
            while not done:
                
                # Get action and advance state.
                action, extra = agent.act(state_torch, explore=train, do_extra=do_extra) # If not in training mode, turn exploration off.
                next_state, reward, done, info = agent.env.step(action)
                next_state_torch = from_numpy(next_state, device=agent.device)
                
                # Perform some agent-specific operations on each timestep if training.
                if train: agent.per_timestep(state_torch, action, reward, next_state_torch, done)

                # Send all information relating to the current timestep to to the observers.
                for o in P["observers"].values(): o.per_timestep(ep, t, state, action, next_state, reward, done, info, extra)

                # Render the environment if applicable.
                if render_this_ep: agent.env.render()

                state = next_state; state_torch = next_state_torch; t += 1
                if by_steps:
                    pbar.update()
                    if pbar.n == pbar.total: force_break = True
                if force_break: break
            if force_break: break

            state = agent.env.reset() # PbrlObserver requires env.reset() here due to video save behaviour.

            # Perform some agent- and observer-specific operations on each episode, which may create logs.
            logs = {}
            if train: logs.update(agent.per_episode())    
            elif hasattr(agent, "per_episode_deploy"): logs.update(agent.per_episode_deploy())
            for o in P["observers"].values(): logs.update(o.per_episode(ep))

            # Send logs to Weights & Biases if applicable.
            if do_wandb: wandb.log(logs)

            # Periodic save-outs of checkpoints (always save after final episode).
            if do_checkpoints and ((ep+1) == P["num_episodes"] or (ep+1) % P["checkpoint_freq"] == 0):
                agent.save(f"{save_dir}/{ep+1}")

        # Clean up.
        if do_wandb: wandb.finish()
        agent.env.close()
    agent.env = env_inside_wrappers

    return run_id, run_name # Return run ID and name for reference.
