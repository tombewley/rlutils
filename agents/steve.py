from .ddpg import DdpgAgent # STEVE inherits from DDPG.
from ._default_hyperparameters import default_hyperparameters
from ..common.dynamics import DynamicsModel
from ..common.utils import col_concat

import numpy as np
import torch
import torch.nn.functional as F


class SteveAgent(DdpgAgent):
    """
    Stochastic ensemble value expansion (STEVE). From:
        "Sample-Efficient Reinforcement Learning with Stochastic Ensemble Value Expansion"
    NOTE: Currently requires reward function to be provided rather than learned.
    """
    def __init__(self, env, hyperparameters):
        assert "reward" in hyperparameters, f"{type(self).__name__} requires a reward function."
        # Overwrite default hyperparameters for DDPG.
        P = default_hyperparameters["ddpg"]
        for k, v in default_hyperparameters["steve"].items(): P[k] = v
        for k, v in hyperparameters.items(): P[k] = v
        DdpgAgent.__init__(self, env, P)
        assert len(self.Q_target) > 1, "Need multiple Q networks to do variance for horizon = 0."
        # Create dynamics model.
        self.P["rollout"]["num_particles"] = self.P["ensemble_size"]
        self.model = DynamicsModel(code=self.P["net_model"], observation_space=self.env.observation_space, action_space=self.env.action_space,
                                     reward_function=self.P["reward"], ensemble_size=self.P["ensemble_size"], rollout_params=self.P["rollout"],
                                     lr=self.P["lr_model"], device=self.device)
        # Small float used to prevent div/0 errors.
        self.eps = np.finfo(np.float32).eps.item()
        # Tracking variables.
        self.random_mode = True
        self.ep_losses_model = []
        self.ep_model_usage = []

    def act(self, state, explore=True, do_extra=False):
        """Either random or DDPG action selection."""
        with torch.no_grad():
            extra = {}
            if self.random_mode: action = self.env.action_space.sample()
            else: action, extra = DdpgAgent.act(self, state, explore, do_extra)
            if do_extra: 
                action_torch = torch.tensor(action, device=self.device).unsqueeze(0)
                extra["next_state_pred"] = self.model.predict(state, action_torch)[0].cpu().numpy()
            return action, extra

    def update_on_batch(self):
        """Use random batches from the replay memory to update the model. Then sample an independent batch
        and use the model for value expansion when updating pi and Q network parameters."""
        if self.total_t % self.P["model_freq"] == 0:
            # Optimise each network in the model ensemble on an independent batch.
            for i in range(self.P["ensemble_size"]):
                states, actions, _, _, next_states = self.memory.sample(self.P["batch_size"], keep_terminal_next=True)
                if states is None: return 
                self.ep_losses_model.append(self.model.update_on_batch(states, actions, next_states, i))
        # TODO: Skip terminal next_states?
        # Sample another batch, this time for training pi and Q, using the model to obtain (hopefully) better Q_targets.
        states, actions, _, nonterminal_mask, next_states = self.memory.sample(self.P["batch_size"], keep_terminal_next=True)
        with torch.no_grad():
            sim_actions = torch.empty((self.P["ensemble_size"], self.model.horizon+1, self.P["batch_size"], actions.shape[1]), device=self.device)
            sim_rewards = torch.zeros((self.P["ensemble_size"], self.model.horizon+1, self.P["batch_size"]), device=self.device)
            sim_rewards[:,0] = self.model.reward_function(states, actions, next_states) # Immediate rewards.
            # Simulate forward dynamics from *next_states* using each network in the ensemble, consulting pi_target for action selection.
            sim_states, sim_actions[:,:-1], sim_rewards[:,1:] = self.model.rollout(next_states, policy=self.pi_target_scaled, ensemble_index=list(range(self.P["ensemble_size"])))
            sim_actions[:,-1] = self.pi_target_scaled(sim_states[:,-1]) # Use pi_target again on the last states.
            # Discount the rewards and take cumulative sum to yield returns up to each horizon.
            gamma_range = torch.tensor([self.P["gamma"]**t for t in range(self.model.horizon+2)], device=self.device).reshape(1,-1,1)
            sim_returns = torch.cumsum(gamma_range[:,:-1] * sim_rewards, dim=1)
            # Finally, add discounted Q values as predicted by each Q_target network.
            Q_targets = torch.cat([(sim_returns + (gamma_range[:,1:] * target_net(col_concat(sim_states, sim_actions)).squeeze(3))
                                   ).unsqueeze(0) for target_net in self.Q_target])
            if False: # Old method for sanity checking
            
                print(Q_targets[:,:,-1])
                
                # Use models to build (hopefully) better Q_targets by simulating forward dynamics.
                Q_targets = torch.zeros((self.P["batch_size"], self.model.horizon+1, self.P["ensemble_size"], len(self.Q_target)))
                # Compute model-free targets.
                rewards = self.model.reward_function(states, actions, next_states).reshape(-1,1)
                next_actions = self.pi_target_scaled(next_states) # Select a' using the target pi network.
                for j, target_net in enumerate(self.Q_target):
                    # Same target for all models at this point.
                    Q_targets[:,0,:,j] = (rewards + self.P["gamma"] * target_net(col_concat(next_states, next_actions))
                    ).expand(self.P["batch_size"], self.P["ensemble_size"])
                for i in range(self.P["ensemble_size"]):
                    # Run a forward simulation for each model.
                    sim_states, sim_actions, sim_returns = next_states, next_actions, rewards.clone()
                    for h in range(1, self.model.horizon+1):
                        # Use model and target pi network to get next states and actions.
                        sim_next_states = self.model.predict(sim_states, sim_actions, ensemble_index=i)
                        sim_next_actions = self.pi_target_scaled(sim_next_states)
                        # Use reward function to get reward for simulated state-action-next-state tuple and add to cumulative return.
                        sim_rewards = self.model.reward_function(sim_states, sim_actions, sim_next_states).reshape(-1,1)
                        assert sim_rewards.shape == (self.P["batch_size"], 1)
                        sim_returns += (self.P["gamma"] ** h) * sim_rewards
                        # Store Q_targets for this horizon.
                        for j, target_net in enumerate(self.Q_target):
                            Q_targets[:,h,i,j] = (sim_returns + ((self.P["gamma"] ** (h+1)) * target_net(col_concat(sim_next_states, sim_next_actions)))
                            ).squeeze()
                        sim_states, sim_actions = sim_next_states, sim_next_actions

                    # print(sim_states)
                Q_targets = Q_targets.permute(3,2,1,0)
                print(Q_targets[:,:,-1])  
                raise Exception()
        # Inverse variance weighting of horizons. 
        var = Q_targets.var(dim=(0, 1)) + self.eps # Prevent div/0 error.
        inverse_var = 1 / var
        normalised_weights = inverse_var / inverse_var.sum(dim=0, keepdims=True)
        self.ep_model_usage.append(1 - normalised_weights[0].mean().item())
        Q_targets = (Q_targets.mean(dim=(0, 1)) * normalised_weights).sum(dim=0)
        # Send Q_targets to DDPG update function and return losses.
        return DdpgAgent.update_on_batch(self, states, actions, Q_targets)

    def per_timestep(self, state, action, reward, next_state, done):
        """Operations to perform on each timestep during training."""
        if self.random_mode and self.total_t >= self.P["num_random_steps"]: 
            self.random_mode = False
            print("Random data collection complete.")
        DdpgAgent.per_timestep(self, state, action, reward, next_state, done, suppress_update=self.random_mode)

    def per_episode(self):
        """Operations to perform on each episode end during training."""
        logs = DdpgAgent.per_episode(self)
        logs["model_loss"] = np.mean(self.ep_losses_model) if self.ep_losses_model else 0.
        logs["model_usage"] = np.mean(self.ep_model_usage) if self.ep_model_usage else 0.
        logs["random_mode"] = int(self.random_mode)
        del self.ep_losses_model[:]; del self.ep_model_usage[:]
        return logs

    def pi_target_scaled(self, states):
        return self._action_scale(self.pi_target(states))
