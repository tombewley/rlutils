from ._generic import Agent
from ..common.memory import ReplayMemory
from ..common.dynamics import DynamicsModel

import torch
import torch.nn.functional as F
from numpy import mean


class SimpleModelBasedAgent(Agent):
    """
    Simple model-based agent for both discrete and continuous action spaces.
    Adapted from the model-based component of the architecture from:
        "Neural Network Dynamics for Model-Based Deep Reinforcement Learning with Model-Free Fine-Tuning"
    """
    def __init__(self, env, hyperparameters):
        assert "reward" in hyperparameters, f"{type(self).__name__} requires a reward function."
        Agent.__init__(self, env, hyperparameters)
        # Create dynamics model.
        self.model = DynamicsModel(code=self.P["net_model"], observation_space=self.env.observation_space, action_space=self.env.action_space,
                                   reward_function=self.P["reward"], probabilistic=self.P["probabilistic"], lr=self.P["lr_model"],
                                   planning_params=self.P["planning"], device=self.device)
        # Create replay memory in two components: one for random transitions one for on-policy transitions.
        self.random_memory = ReplayMemory(self.P["num_random_steps"])
        if not self.P["random_mode_only"]:
            self.memory = ReplayMemory(self.P["replay_capacity"])
            self.batch_split = (round(self.P["batch_size"] * self.P["batch_ratio"]), round(self.P["batch_size"] * (1-self.P["batch_ratio"])))
        # Tracking variables.
        self.random_mode = True
        self.total_t = 0 # Used for model_freq.
        self.ep_losses = []

    def act(self, state, explore=True, do_extra=False):
        """Either random or model-based action selection."""
        with torch.no_grad():
            do_extra=True
            extra = {}
            if self.random_mode: 
                action = self.env.action_space.sample()
                if do_extra: 
                    action_torch = torch.tensor(action, device=self.device).unsqueeze(0)
                    extra["next_state_pred"] = self.model.predict(state, action_torch)[0].cpu().numpy()
            else: 
                states, actions, returns, mean, std = self.model.plan(state)
                best = torch.argmax(returns[-1]) # Look at final planning iteration only
                action = actions[-1,best,0] # Take first action only
                action = action.cpu().numpy() if self.model.continuous_actions else action.item()
                if do_extra: 
                    extra["g_pred"] = returns[-1,best].item()
                    extra["next_state_pred"] = states[-1,best,1].cpu().numpy()
            return action, extra

    def update_on_batch(self):
        """Use a random batch from the replay memory to update the model network parameters."""
        if self.random_mode: # During random mode, just sample from random memory.   
            states, actions, _, _, next_states = self.random_memory.sample(self.P["batch_size"], keep_terminal_next=True)
            if states is None: return 
        else: # After random mode, sample from both memories according to self.batch_split.
            states, actions, _, _, next_states = self.memory.sample(self.batch_split[0], keep_terminal_next=True)
            if states is None: return 
            states_r, actions_r, _, _, next_states_r = self.random_memory.sample(self.batch_split[1], keep_terminal_next=True)
            assert states_r is not None, "Random mode not long enough!"
            states = torch.cat((states, states_r), dim=0)
            actions = torch.cat((actions, actions_r), dim=0)
            next_states = torch.cat((next_states, next_states_r), dim=0)        
        if not self.P["probabilistic"]:
            # Update model in the direction of the true state derivatives using MSE loss.
            loss = F.mse_loss(self.model.predict(states, actions), next_states)
        else:
            # Update model using Gaussian negative log likelihood loss (see PETS paper equation 1).
            mu, log_std = self.model.predict(states, actions, params=True); log_var = 2 * log_std
            loss = (F.mse_loss(states + mu, next_states, reduction="none") * (-log_var).exp() + log_var).mean() 
            # TODO: Add a small regularisation penalty to prevent growth of variance range.
            # loss += 0.01 * (self.max_log_var.sum() - self.min_log_var.sum()) 
        self.model.net.optimise(loss)
        return loss.item()

    def per_timestep(self, state, action, reward, next_state, done):
        """Operations to perform on each timestep during training."""
        if not self.P["random_mode_only"] and self.random_mode and len(self.random_memory) >= self.P["num_random_steps"]: 
            self.random_mode = False
            print("Random data collection complete.")
        if self.random_mode: self.random_memory.add(state, action, reward, next_state, done)
        else: self.memory.add(state, action, reward, next_state, done)
        self.total_t += 1
        if self.P["model_freq"] > 0 and self.total_t % self.P["model_freq"] == 0:
            loss = self.update_on_batch()
            if loss is not None: self.ep_losses.append(loss)

    def per_episode(self):
        """Operations to perform on each episode end during training."""
        if self.ep_losses: mean_loss = mean(self.ep_losses)
        else: mean_loss = 0.
        del self.ep_losses[:]
        return {"model_loss": mean_loss, "random_mode": int(self.random_mode)}
