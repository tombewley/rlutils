from ._generic import Agent
from ..common.memory import ReplayMemory
from ..common.dynamics import DynamicsModel
from ..common.utils import truncated_normal, from_numpy

import torch
from numpy import array, float32, mean


class PetsAgent(Agent):
    """
    Probabilistic ensembles with trajectory sampling (PETS). From:
        Chua, Kurtland, Roberto Calandra, Rowan McAllister, and Sergey Levine. 
        "Deep reinforcement learning in a handful of trials using probabilistic dynamics models." 
        Advances in Neural Information Processing Systems 31 (2018).

    Also likely to retain some details of the model-based component of MBMF. From:
        "Neural Network Dynamics for Model-Based Deep Reinforcement Learning with Model-Free Fine-Tuning"
    """
    def __init__(self, env, hyperparameters):
        assert "reward_function" in hyperparameters, f"{type(self).__name__} requires a reward function."
        Agent.__init__(self, env, hyperparameters)
        self.rng = torch.Generator(device=self.device)
        # Create dynamics model, optionally loading a pretrained ensemble of nets.
        if "pretrained_model" in self.P: nets, code, ensemble_size, probabilistic = self.P["pretrained_model"], None, None, None
        else: nets, code, ensemble_size, probabilistic = None, self.P["net_model"], self.P["ensemble_size"], self.P["probabilistic"]
        if "termination_function" in self.P: term_func = self.P["termination_function"]; assert "cost_function" not in self.P, "Incompatible!"
        else:                                term_func = None
        self.model = DynamicsModel(nets=nets, code=code, observation_space=self.env.observation_space, action_space=self.env.action_space,
                                   reward_function=self.P["reward_function"], termination_function=term_func,
                                   normaliser=self.P["input_normaliser"], probabilistic=probabilistic, delta=self.P["delta_dynamics"],
                                   lr=self.P["lr_model"], ensemble_size=ensemble_size, rollout_params=self.P["rollout"], device=self.device)
        # Action space bounds, needed for CEM.
        self.action_space_low = torch.tensor(self.env.action_space.low, device=self.device)
        self.action_space_high = torch.tensor(self.env.action_space.high, device=self.device)
        # Create replay memory in two components: one for random transitions one for on-policy transitions.
        self.random_mode = self.P["num_random_steps"] > 0
        if self.random_mode: self.random_memory = ReplayMemory(self.P["num_random_steps"])
        self.memory = ReplayMemory(self.P["replay_capacity"])
        self.batch_split = (round(self.P["batch_size"] * self.P["batch_ratio"]), round(self.P["batch_size"] * (1-self.P["batch_ratio"])))
        # Initialise action prior using action space bounds.
        self.reset_action_prior()
        # Tracking variables.
        self.total_t = 0
        self.ep_action_stds, self.ep_losses = [], []

    def seed(self, seed):
        self.model.seed(seed)
        self.rng.manual_seed(seed)

    def reset_action_prior(self):
        self.action_prior_mean = self.act_b[None,None].tile((self.model.horizon,1,1))
        self.action_prior_std  = self.act_k[None,None].tile((self.model.horizon,1,1))

    def force_action(self, t, action):
        assert array(action, dtype=float32) in self.env.action_space
        self.action_prior_mean[t] = from_numpy(action, device=self.device)
        self.action_prior_std[t] = 0

    def act(self, state, explore=True, do_extra=False):
        """Either random or model-based action selection."""
        with torch.no_grad():
            extra = {}
            if explore and self.random_mode: action = self.env.action_space.sample()
            else:
                # CEM: use model and reward function to generate and evaluate action sequences sampled from a truncated normal.
                # After each iteration, refine the parameters of the truncated normal based on a subset of elites and resample.
                action_dim = self.env.action_space.shape[0]
                mean    = torch.empty((self.P["cem_iterations"],                          self.model.horizon, 1, action_dim), device=self.device)
                std     = torch.empty((self.P["cem_iterations"],                          self.model.horizon, 1, action_dim), device=self.device)
                actions = torch.empty((self.P["cem_iterations"], self.P["cem_particles"], self.model.horizon, 1, action_dim), device=self.device)
                returns = torch.zeros((self.P["cem_iterations"], self.P["cem_particles"]                                   ), device=self.device)
                gamma_range = torch.tensor([self.P["gamma"]**t for t in range(self.model.horizon)], device=self.device).reshape(1,-1,1)
                # Adopt current prior for the first iteration.
                mean[0], std[0] = self.action_prior_mean, self.action_prior_std
                if do_extra: states = []
                for i in range(self.P["cem_iterations"]):
                    if i > 0:
                        # Update sampling distribution using weighted mean/std of elites from previous iteration,
                        # as in Model Predictive Path Integral (MPPI) control.
                        # See "Temporal Difference Learning for Model Predictive Control", Section 3.
                        elite_weights = torch.exp(self.P["cem_temperature"] * (returns[i-1,elites] - returns[i-1,elites[0]]))
                        elite_weights[returns[i-1,elites].isneginf()] = 0. # It's okay for some returns to be -inf due to constraints...
                        elite_weights_sum = elite_weights.sum()
                        assert elite_weights_sum > 0                       # ...but they can't all be!
                        mean_elite = torch.tensordot(actions[i-1,elites], elite_weights, ([0],[0])) / elite_weights_sum
                        std_elite = torch.sqrt(torch.tensordot((actions[i-1,elites] - mean_elite)**2, elite_weights, ([0],[0])) / elite_weights_sum)
                        mean[i] = (1 - self.P["cem_alpha"]) * mean[i-1] + self.P["cem_alpha"] * mean_elite
                        std[i] = (1 - self.P["cem_alpha"]) * std[i-1] + self.P["cem_alpha"] * std_elite
                    # Sample action sequences from truncated normal parameterised by mean/std and action space bounds.
                    actions[i] = truncated_normal(actions[i], mean=mean[i], std=std[i], a=self.action_space_low, b=self.action_space_high, rng=self.rng)
                    # The above will put NaN values anywhere that std[i] == 0. In these cases, overwrite with mean[i].
                    zero_std = std[i] == 0; actions[i,:,zero_std] = mean[i,zero_std]
                    if i == 0 and self.P["cem_initial_inertia"]:
                        # Optionally use inertia to smooth actions on first iteration; may aid exploration in some envs.
                        k = self.P["cem_initial_inertia"]
                        for t in range(1, actions[i].shape[1]):
                            actions[i,:,t] = k * actions[i,:,t-1] + (1 - k) * actions[i,:,t]
                    # Propagate action sequences through the model and compute returns.
                    s, _, rewards = self.model.rollout(state, actions=actions[i], ensemble_index="ts1_a")
                    returns[i] = (gamma_range * rewards).sum(axis=1).squeeze()
                    # If a cost function is provided in addition to a reward function, add costs to returns.
                    # NOTE: the cost function does not need to be Markovian.
                    if "cost_function" in self.P: returns[i] += self.P["cost_function"](s[:,:-1], actions[i], s[:,1:])
                    elites = returns[i].topk(self.P["cem_elites"]).indices
                    if do_extra: states.append(s)
                # Take first action from best elite.
                action = actions[-1,elites[0],0].squeeze(0)
                action = action.cpu().numpy() if self.continuous_actions else action.item()
                self.ep_action_stds.append((std[-1,0] / self.act_k).mean().item())
                # Optionally adopt time-shifted mean action for t+1 onwards to use as a warm-start prior for the next timestep.
                if self.P["cem_warm_start"]: self.action_prior_mean[:-1] = mean[-1,1:]
                if do_extra:
                    extra["mean"], extra["std"], extra["states"], extra["actions"] = \
                    mean.squeeze(), std.squeeze(), torch.stack(states).squeeze(), actions.squeeze()
            return action, extra

    def update_on_batch(self):
        """Use a random batch from the replay memory to update the model network parameters."""
        # Optimise each network in the model ensemble on an independent batch.
        for i in range(len(self.model.nets)):
            if self.random_mode: # During random mode, just sample from random memory.   
                states, actions, _, _, next_states = self.random_memory.sample(self.P["batch_size"], keep_terminal_next=True)
                if states is None: return 
            else: # After random mode, sample from both memories according to self.batch_split.
                states, actions, _, _, next_states = self.memory.sample(self.batch_split[0], keep_terminal_next=True)
                if states is None: return 
                if self.batch_split[1] > 0:
                    states_r, actions_r, _, _, next_states_r = self.random_memory.sample(self.batch_split[1], keep_terminal_next=True)
                    assert states_r is not None, "Random mode not long enough!"
                    states = torch.cat((states, states_r), dim=0)
                    actions = torch.cat((actions, actions_r), dim=0)
                    next_states = torch.cat((next_states, next_states_r), dim=0)
            self.ep_losses.append(self.model.update_on_batch(states, actions, next_states, i))

    def per_timestep(self, state, action, reward, next_state, done):
        """Operations to perform on each timestep during training."""
        if self.random_mode and len(self.random_memory) >= self.P["num_random_steps"]:
            self.random_mode = False
            print("Random data collection complete.")
        if self.random_mode: self.random_memory.add(state, action, reward, next_state, done)
        else: self.memory.add(state, action, reward, next_state, done)
        self.total_t += 1
        if self.P["model_freq"] > 0 and self.total_t % self.P["model_freq"] == 0: self.update_on_batch()

    def per_episode(self):
        """Operations to perform on each episode end during training."""
        logs = {"model_loss": mean(self.ep_losses) if self.ep_losses else 0.,
                "random_mode": int(self.random_mode),
                "horizon": self.model.horizon,
                "next_action_std": mean(self.ep_action_stds) if self.ep_action_stds else 0.}
        del self.ep_action_stds[:], self.ep_losses[:]
        self.reset_action_prior()
        return logs

    def save(self, path, clear_memory=True):
        print("Saving dynamics model only")
        torch.save(self.model.nets, f"{path}.dynamics")
