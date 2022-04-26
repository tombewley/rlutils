from .networks import SequentialNetwork
from .utils import col_concat, reparameterise

import torch
from random import choice
from gym.spaces.box import Box


"""
TODO:

- Smarter/manual initialisation of truncated normal parameters
- With discrete actions, better to have per-action output nodes rather than providing action integer as an input.
- Fixed dim indices to prevent unnecessary prediction effort

"""


class DynamicsModel:
    """
    Optionally implements probabilistic dynamics using the reparameterisation trick.
    """
    def __init__(self, observation_space, action_space, reward_function, termination_function, lr, rollout_params, device, nets=None, code=None, ensemble_size=None, probabilistic=False):
        self.reward_function = reward_function
        self.termination_function = termination_function
        self.probabilistic = probabilistic
        self.P = rollout_params
        assert type(action_space) == Box, "CEM doesn't work with discrete actions, and need one-hot encoding for model"
        if nets is not None: # Load pretrained ensemble of nets.
            for net in nets:
                for g in net.optimiser.param_groups: g["lr"] = lr
            self.nets = nets
        else: # Create new ensemble of nets.
            self.nets = [SequentialNetwork(code=code, input_space=[observation_space, action_space],
                                        output_size=observation_space.shape[0]*(2 if self.probabilistic else 1),
                                        normaliser="box_bounds", lr=lr, device=device) # NOTE: Using box_bounds normalisation.
                        for _ in range(ensemble_size)]
        self.horizon = self.P["horizon_params"][1] # Initial planning horizon (NOTE: indexing assumption).
        self.action_dim = action_space.shape[0]
        # Weight model loss function by bounds of observation space.
        self.loss_weights = torch.tensor(1. / (observation_space.high - observation_space.low), device=device)
        # NOTE: Currently MBRL-Lib says fixed bounds "work better" than learnt ones. Using values from there (note std instead of var).
        if self.probabilistic: self.log_std_clamp = ("hard", -20, 2) # ("soft", -20, 2)
        # Tracking variables.
        self.num_updates = 0

    def predict(self, states, actions, ensemble_index, params=False):
        """
        Predict the next state for an array of state-action pairs.
        """
        states_and_actions = col_concat(states, actions)
        if type(ensemble_index) == int: # Use the same ensemble member for all state-action pairs.
            ds = self.nets[ensemble_index](states_and_actions)
        elif type(ensemble_index) == list: # Use a specified member for each pair.
            assert len(ensemble_index) == states_and_actions.shape[0]
            ds = torch.cat([self.nets[ensemble_index[i]](sa).unsqueeze(0) for i, sa in enumerate(states_and_actions)], dim=0)
        elif ensemble_index == "ts1_a": # Uniform-randomly sample a common member to use for all pairs.
            ds = choice(self.nets)(states_and_actions)
        elif ensemble_index == "ts1_b": # Uniform-randomly sample a member to use for each pair.
            ds = torch.cat([choice(self.nets)(sa).unsqueeze(0) for sa in states_and_actions], dim=0)
        elif ensemble_index == "all": # Use the entire ensemble for all pairs.
            print("TODO: test that this works with self.probabilistic=True")
            ds = torch.cat([net(states_and_actions).unsqueeze(2) for net in self.nets], dim=2)
        # If using a probabilistic dynamics model, employ the reparameterisation trick.
        if self.probabilistic: 
            if params: return reparameterise(ds, clamp=self.log_std_clamp, params=True) # Return mean and log standard deviation.
            else: ds = reparameterise(ds, clamp=self.log_std_clamp).rsample() 
        return states + ds

    def rollout(self, states_init, ensemble_index, policy=None, actions=None):
        """
        Starting at states_init, rollout either a callable policy or predefined action sequences.
        NOTE: PETS paper seems to use multiple particles per action sequence.
        """
        batch_size = states_init.shape[0]
        if actions is not None:
            assert torch.is_tensor(actions) and actions.shape[2:] == (batch_size, self.action_dim)
            using_policy, (num_particles, horizon) = False, actions.shape[:2]
        elif policy is not None:
            assert callable(policy)
            using_policy, num_particles, horizon = True, self.P["num_particles"], self.horizon
            actions = torch.empty((num_particles, horizon,   batch_size, self.action_dim     ), device=states_init.device)
        else: raise ValueError("Must provide either policy or actions.")
        states      = torch.empty((num_particles, horizon+1, batch_size, states_init.shape[1]), device=states_init.device)
        rewards     = torch.zeros((num_particles, horizon,   batch_size                      ), device=states_init.device)
        if self.termination_function is not None:
            dones   = torch.zeros((num_particles,  horizon,  batch_size                      ), device=states_init.device, dtype=bool)
        states[:,0] = states_init
        for t in range(horizon):
            if using_policy: actions[:,t] = policy(states[:,t]) # If using a policy, action selection is closed-loop.
            states[:,t+1] = self.predict(states[:,t], actions[:,t], ensemble_index)
            rewards[:,t]  = self.reward_function(states[:,t], actions[:,t], states[:,t+1])
            if self.termination_function is not None and t < (horizon-1):
                dones[:,t+1] = self.termination_function(states[:,t], actions[:,t], states[:,t+1])
        # Retroactively zero out post-termination rewards. NOTE: Simple but quite wasteful.
        if self.termination_function is not None: rewards[torch.cumsum(dones, dim=1) > 0] = 0.
        return states, actions, rewards

    def update_on_batch(self, states, actions, next_states, ensemble_index):
        """
        Update one member of the ensemble using a batch of transitions.
        """

        # TODO: Use multiple gradient steps and holdout_ratio
        # https://github.com/Xingyu-Lin/mbpo_pytorch/blob/main/model.py

        if not self.probabilistic:
            # Update model in the direction of the true state derivatives using weighted MSE loss.
            loss = ((self.loss_weights * (self.predict(states, actions, ensemble_index=ensemble_index) - next_states)) ** 2).mean()
        else:
            raise NotImplementedError("Haven't implemented weighting (careful with variance!)")
            # Update model using Gaussian negative log likelihood loss (see PETS paper equation 1).
            mu, log_std = self.predict(states, actions, ensemble_index=ensemble_index, params=True)
            log_var = 2 * log_std
            loss = (F.mse_loss(states + mu, next_states, reduction="none") * (-log_var).exp() + log_var).mean() 
            # TODO: Add a small regularisation penalty to prevent growth of variance range.
            # loss += 0.01 * (self.max_log_var.sum() - self.min_log_var.sum()) 
        self.nets[ensemble_index].optimise(loss)
        return loss.item()

    def _update_horizon(self):
        if self.P["horizon_params"][0] == "constant": pass
        elif self.P["horizon_params"][0] == "linear":
            _, x, y, (a, b) = self.P["horizon_params"]
            self.horizon = int(round(min(max(x + (((self.num_updates - a) / (b - a)) * (y - x)), x), y)))
        else: raise NotImplementedError()
