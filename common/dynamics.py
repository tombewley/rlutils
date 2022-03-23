from .networks import SequentialNetwork
from .utils import col_concat, reparameterise, truncated_normal

import torch


"""
TODO:

- Smarter/manual initialisation of truncated normal parameters
- Batch/hard normalisation of state dimensions (normalise layer?)
- TS1 uniformly resample a dynamics model per rollout timestep
- With discrete actions, better to have per-action output nodes rather than providing action integer as an input.

"""


class DynamicsModel:
    """
    Optionally implements probabilistic dynamics using the reparameterisation trick.
    """
    def __init__(self, code, observation_space, action_space, reward_function, probabilistic, lr, planning_params, device):
        self.reward_function = reward_function
        self.probabilistic = probabilistic
        self.P = planning_params
        self.continuous_actions = len(action_space.shape) > 0
        assert self.continuous_actions, "CEM doesn't work with discrete actions"
        if len(observation_space.shape) > 1: raise NotImplementedError()
        else: state_dim, action_dim = observation_space.shape[0], (action_space.shape[0] if self.continuous_actions else 1) 
        self.net = SequentialNetwork(code=code, input_shape=state_dim+action_dim,
                                     output_size=state_dim*(2 if self.probabilistic else 1), lr=lr, device=device)
        self.action_space_low = torch.tensor(action_space.low, device=device)
        self.action_space_high = torch.tensor(action_space.high, device=device)
        # NOTE: Currently MBRL-Lib says fixed bounds "work better" than learnt ones. Using values from there (note std instead of var).
        if self.probabilistic: self.log_std_clamp = ("hard", -20, 2) # ("soft", -20, 2)

    def predict(self, states, actions, params=False):
        """
        Predict the next state for an array of state-action pairs.
        """
        ds = self.net(col_concat(states, actions if self.continuous_actions else actions.unsqueeze(1)))
        # If using a probabilistic dynamics model, employ the reparameterisation trick.
        if self.probabilistic: 
            if params: return reparameterise(ds, clamp=self.log_std_clamp, params=True) # Return mean and log standard deviation.
            else: ds = reparameterise(ds, clamp=self.log_std_clamp).rsample() 
        return states + ds

    def plan(self, state_init):
        """
        Use model and reward function to generate and evaluate action sequences sampled from a truncated normal.
        After each iteration, refine the parameters of the truncated normal based on a subset of elites and resample.
        """
        # Create empty tensors.
        action_dim = self.action_space_low.shape[0]
        states  = torch.empty((self.P["num_iterations"], self.P["num_particles"], self.P["horizon"]+1, state_init.shape[1]), device=state_init.device)
        actions = torch.empty((self.P["num_iterations"], self.P["num_particles"], self.P["horizon"],   action_dim         ), device=state_init.device)
        returns = torch.zeros((self.P["num_iterations"], self.P["num_particles"]),                                           device=state_init.device)
        mean    = torch.empty((self.P["num_iterations"],                          self.P["horizon"],   action_dim         ), device=state_init.device)
        std     = torch.empty((self.P["num_iterations"],                          self.P["horizon"],   action_dim         ), device=state_init.device)
        # Initiate state at t = 0 and mean/std at i = 0.
        states[:,:,0] = state_init

        #### TODO: ####
        mean[0] = torch.zeros((self.P["horizon"], action_dim))
        std[0] = 2*torch.ones((self.P["horizon"], action_dim))
        ###############

        for i in range(self.P["num_iterations"]):
            if i > 0:
                # Update mean/std using elites from previous iteration.
                elites = returns[i-1].topk(self.P["num_elites"]).indices
                std_elite, mean_elite = torch.std_mean(actions[i-1,elites], dim=0, unbiased=False)
                mean[i] = self.P["alpha"] * mean[i-1] + (1 - self.P["alpha"]) * mean_elite
                std[i] = self.P["alpha"] * std[i-1] + (1 - self.P["alpha"]) * std_elite
            # Sample action sequences from truncated normal parameterised by mean/std and action space bounds.
            actions[i] = truncated_normal(actions[i], mean=mean[i], std=std[i], a=self.action_space_low, b=self.action_space_high)
            # Evaluate action sequences using dynamics model and reward function.
            for t in range(self.P["horizon"]):
                states[i,:,t+1] = self.predict(states[i,:,t], actions[i,:,t])
                returns[i] += (self.P["gamma"] ** t) * self.reward_function(states[i,:,t], actions[i,:,t], states[i,:,t+1])
        return states, actions, returns, mean, std
