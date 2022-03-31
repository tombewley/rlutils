from ._generic import Agent
from ..common.memory import ReplayMemory
from ..common.dynamics import DynamicsModel
from ..common.utils import truncated_normal

import torch
from numpy import mean


class PetsAgent(Agent):
    """
    Probabilistic ensembles with trajectory sampling (PETS). From:
        Chua, Kurtland, Roberto Calandra, Rowan McAllister, and Sergey Levine. 
        "Deep reinforcement learning in a handful of trials using probabilistic dynamics models." 
        Advances in Neural Information Processing Systems 31 (2018).

    Also likely to retain some details of the model-based component from:
        "Neural Network Dynamics for Model-Based Deep Reinforcement Learning with Model-Free Fine-Tuning"
    """
    def __init__(self, env, hyperparameters):
        assert "reward" in hyperparameters, f"{type(self).__name__} requires a reward function."
        Agent.__init__(self, env, hyperparameters)
        # Create dynamics model.
        self.model = DynamicsModel(code=self.P["net_model"], observation_space=self.env.observation_space, action_space=self.env.action_space,
                                   reward_function=self.P["reward"], probabilistic=self.P["probabilistic"], lr=self.P["lr_model"],
                                   ensemble_size=self.P["ensemble_size"], rollout_params=self.P["rollout"], device=self.device)
        # Action space bounds, needed for CEM.
        self.action_space_low = torch.tensor(self.env.action_space.low, device=self.device)
        self.action_space_high = torch.tensor(self.env.action_space.high, device=self.device)
        # Create replay memory in two components: one for random transitions one for on-policy transitions.
        self.random_memory = ReplayMemory(self.P["num_random_steps"])
        self.memory = ReplayMemory(self.P["replay_capacity"])
        self.batch_split = (round(self.P["batch_size"] * self.P["batch_ratio"]), round(self.P["batch_size"] * (1-self.P["batch_ratio"])))
        # Tracking variables.
        self.random_mode = True
        self.total_t = 0 # Used for model_freq.
        self.ep_losses = []

    def act(self, state, explore=True, do_extra=False):
        """Either random or model-based action selection."""
        with torch.no_grad():
            extra = {}
            if self.random_mode: 
                action = self.env.action_space.sample()
                if do_extra: 
                    action_torch = torch.tensor(action, device=self.device).unsqueeze(0)
                    extra["next_state_pred"] = self.model.predict(state, action_torch)[0].cpu().numpy()
            else:
                # CEM: use model and reward function to generate and evaluate action sequences sampled from a truncated normal.
                # After each iteration, refine the parameters of the truncated normal based on a subset of elites and resample.
                action_dim = self.env.action_space.shape[0]
                mean    = torch.empty((self.P["cem_iterations"], self.model.horizon, 1, action_dim), device=self.device)
                std     = torch.empty((self.P["cem_iterations"], self.model.horizon, 1, action_dim), device=self.device)
                # NOTE: Initialise distribution parameters based on action space bounds.
                mean[0], std[0] = self.act_b, self.act_k
                actions = torch.empty((self.P["cem_iterations"], self.P["cem_particles"], self.model.horizon, 1, action_dim), device=self.device)
                returns = torch.zeros((self.P["cem_iterations"], self.P["cem_particles"]                                   ), device=self.device)
                gamma_range = torch.tensor([self.P["gamma"]**t for t in range(self.model.horizon)], device=self.device).reshape(1,-1,1)
                for i in range(self.P["cem_iterations"]):
                    if i > 0:
                        # Update sampling distribution using elites from previous iteration.
                        std_elite, mean_elite = torch.std_mean(actions[i-1,elites], dim=0, unbiased=False)
                        mean[i] = (1 - self.P["cem_alpha"]) * mean[i-1] + self.P["cem_alpha"] * mean_elite
                        std[i] = (1 - self.P["cem_alpha"]) * std[i-1] + self.P["cem_alpha"] * std_elite
                    # Sample action sequences from truncated normal parameterised by mean/std and action space bounds.
                    actions[i] = truncated_normal(actions[i], mean=mean[i], std=std[i], a=self.action_space_low, b=self.action_space_high)
                    # Propogate action sequences through the model.
                    states, _, rewards = self.model.rollout(state, actions=actions[i], ensemble_index="ts1")
                    returns[i] = (gamma_range * rewards).sum(axis=1).squeeze()
                    elites = returns[i].topk(self.P["cem_elites"]).indices
                best = elites[0]
                action = actions[-1,best,0] # Take first action only
                action = action.cpu().numpy()[0] if self.continuous_actions else action.item()
                if do_extra: 
                    extra["g_pred"] = returns[-1,best].item()
                    extra["next_state_pred"] = states[best,1].cpu().numpy()
            return action, extra

    def update_on_batch(self):
        """Use a random batch from the replay memory to update the model network parameters."""
        # Optimise each network in the model ensemble on an independent batch.
        for i in range(self.P["ensemble_size"]):
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
        mean_loss = mean(self.ep_losses) if self.ep_losses else 0.
        del self.ep_losses[:]
        return {"model_loss": mean_loss, "random_mode": int(self.random_mode)}
