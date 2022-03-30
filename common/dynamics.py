from .networks import SequentialNetwork, BoxNormalise
from .utils import col_concat, reparameterise, truncated_normal

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
    def __init__(self, code, observation_space, action_space, reward_function, lr, ensemble_size, planning_params, device, probabilistic=False):
        self.reward_function = reward_function
        self.probabilistic = probabilistic
        self.P = planning_params
        assert type(action_space) == Box, "CEM doesn't work with discrete actions, and need one-hot encoding for model"
        self.nets = [SequentialNetwork(code=code, input_space=[observation_space, action_space],
                                     output_size=observation_space.shape[0]*(2 if self.probabilistic else 1),
                                     normaliser="box_bounds", lr=lr, device=device) # NOTE: Using box_bounds normalisation.
                     for _ in range(ensemble_size)]
        # Initial planning horizon.
        self.horizon = self.P["horizon_params"][1] # NOTE: indexing assumption.
        # Parameters for state and action scaling.
        self.state_loss_weights = torch.tensor(1. / (observation_space.high - observation_space.low), device=device)
        self.action_space_low = torch.tensor(action_space.low, device=device)
        self.action_space_high = torch.tensor(action_space.high, device=device)
        self.act_k = (self.action_space_high - self.action_space_low) / 2.
        self.act_b = (self.action_space_high + self.action_space_low) / 2.
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
        elif ensemble_index == "ts1": # Uniform-randomly sample a member to use for all pairs.
            # TODO: verify that this is the correct approach.
            ds = choice(self.nets)(states_and_actions)
        elif ensemble_index == "all": # Use the entire ensemble for all pairs.
            print("TODO: test that this works with self.probabilistic=True")
            ds = torch.cat([net(states_and_actions).unsqueeze(2) for net in self.nets], dim=2)
        # If using a probabilistic dynamics model, employ the reparameterisation trick.
        if self.probabilistic: 
            if params: return reparameterise(ds, clamp=self.log_std_clamp, params=True) # Return mean and log standard deviation.
            else: ds = reparameterise(ds, clamp=self.log_std_clamp).rsample() 
        return states + ds

    def plan(self, states_init, policy, ensemble_index):
        """
        Use model and reward function to generate and evaluate action sequences sampled from a truncated normal.
        After each iteration, refine the parameters of the truncated normal based on a subset of elites and resample.
        """
        # Create empty tensors. TODO: Initialise just once to save memory allocation time
        batch_size, action_dim = states_init.shape[0], self.action_space_low.shape[0]
        states  = torch.empty((self.P["num_iterations"], self.P["num_particles"], self.horizon+1, batch_size, states_init.shape[1]), device=states_init.device)
        actions = torch.empty((self.P["num_iterations"], self.P["num_particles"], self.horizon,   batch_size, action_dim         ), device=states_init.device)
        rewards = torch.zeros((self.P["num_iterations"], self.P["num_particles"], self.horizon,   batch_size                     ), device=states_init.device)
        # Initiate state at t = 0 and mean/std at i = 0.
        states[:,:,0] = states_init

        assert policy == "cem" or callable(policy) # TODO: Make more generic
        if policy == "cem":
            assert batch_size == 1
            #### TODO: ####
            mean    = torch.empty((self.P["num_iterations"],                          self.horizon,   batch_size, action_dim         ), device=states_init.device)
            std     = torch.empty((self.P["num_iterations"],                          self.horizon,   batch_size, action_dim         ), device=states_init.device)
            mean[0] = torch.zeros((self.horizon, batch_size, action_dim))
            std[0] = 2*torch.ones((self.horizon, batch_size, action_dim))
            # gamma_range = torch.tensor([self.P["gamma"]**(t+1) for t in range(self.model.P["horizon"]+1)]).reshape(1,-1,1)
            ###############

        elif ensemble_index == "tsinf":
            raise NotImplementedError("Allocate a member to each particle at the start")

        for i in range(self.P["num_iterations"]):
            if policy == "cem":
                if i > 0:
                    # Update mean/std using elites from previous iteration.
                    raise NotImplementedError("Use gamma_range")
                    elites = returns[i-1,:,-1,0].topk(self.P["num_elites"]).indices
                    std_elite, mean_elite = torch.std_mean(actions[i-1,elites], dim=0, unbiased=False)
                    mean[i] = (1 - self.P["alpha"]) * mean[i-1] + self.P["alpha"] * mean_elite
                    std[i] = (1 - self.P["alpha"]) * std[i-1] + self.P["alpha"] * std_elite
                # Sample action sequences from truncated normal parameterised by mean/std and action space bounds.
                actions[i] = truncated_normal(actions[i], mean=mean[i], std=std[i], a=self.action_space_low, b=self.action_space_high)
            # Evaluate action sequences using dynamics model and reward function.
            for t in range(self.horizon):
                if callable(policy): actions[i,:,t] = policy(states[i,:,t])
                states[i,:,t+1] = self.predict(states[i,:,t], actions[i,:,t], ensemble_index)
                rewards[i,:,t]  = self.reward_function(states[i,:,t], actions[i,:,t], states[i,:,t+1])
        return states, actions, rewards

    def update_on_batch(self, states, actions, next_states, ensemble_index):
        """
        Update one member of the ensemble using a batch of transitions.
        """

        # TODO: Use multiple gradient updates and holdout_ratio
        # https://github.com/Xingyu-Lin/mbpo_pytorch/blob/main/model.py

        if not self.probabilistic:
            # Update model in the direction of the true state derivatives using weighted MSE loss.
            loss = (self.state_loss_weights * (self.predict(states, actions, ensemble_index=ensemble_index) - next_states) ** 2).mean()
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
