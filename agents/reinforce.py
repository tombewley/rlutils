from ._generic import Agent
from ..common.networks import SequentialNetwork

import numpy as np
import torch
from torch.distributions import Categorical
import torch.nn.functional as F


class ReinforceAgent(Agent):
    """
    REINFORCE / vanilla policy gradient for discrete action spaces.
    """
    def __init__(self, env, hyperparameters):
        Agent.__init__(self, env, hyperparameters)
        self.eps = np.finfo(np.float32).eps.item() # Small float used to prevent div/0 errors.
        # Create pi network (and V if using advantage baselining).
        self.pi = SequentialNetwork(code=self.P["net_pi"], input_space=[self.env.observation_space], output_size=self.env.action_space.n,
                                    normaliser=self.P["input_normaliser"], lr=self.P["lr_pi"], device=self.device)
        if self.P["baseline"] == "adv":
            self.V = SequentialNetwork(code=self.P["net_V"], input_space=[self.env.observation_space], output_size=1,
                                       normaliser=self.P["input_normaliser"], lr=self.P["lr_V"], device=self.device)
            self.ep_values = []
        else: self.V = None
        # Tracking variables.
        self.ep_rewards = []
        self.ep_log_probs = []

    def act(self, state, explore=True, do_extra=False):
        """Probabilistic action selection *without* torch.no_grad() to allow backprop later."""
        action_probs = self.pi(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        self.ep_log_probs.append(dist.log_prob(action).squeeze())
        extra = {"pi": action_probs.squeeze().cpu().detach().numpy()} if do_extra else {}
        if self.V is not None:
            value = self.V(state).squeeze()
            self.ep_values.append(value)
            if do_extra: extra["V"] = value.item()
        return action.item(), extra

    def update_on_episode(self):
        """Use the latest episode of experience to update the policy (and value) network parameters."""
        assert len(self.ep_rewards) == len(self.ep_log_probs)
        # Loop backwards through the episode to compute Monte Carlo returns.
        g, returns = 0, []
        for reward in self.ep_rewards[::-1]:
            g = reward + (self.P["gamma"] * g)
            returns.insert(0, g)
        returns = torch.tensor(returns, device=self.device)
        log_probs = torch.stack(self.ep_log_probs)
        if self.V is not None: 
            # Update value in the direction of advantage using Huber loss.
            values = torch.stack(self.ep_values)
            value_loss = F.smooth_l1_loss(values, returns)
            self.V.optimise(value_loss)
        else: values, value_loss = None, None
        # Update policy in the direction of log_prob(a) * delta.
        policy_loss = (-log_probs * self.baseline(returns, values)).mean()
        self.pi.optimise(policy_loss)
        return policy_loss.item(), value_loss.item()

    def baseline(self, returns, values):
        """Apply baselining to returns to improve update stability."""
        if   self.P["baseline"] == "off": return returns # No baselining.
        elif self.P["baseline"] == "Z":   return (returns - returns.mean()) / (returns.std() + self.eps) # Z-normalisation.
        elif self.P["baseline"] == "adv": return (returns - values).detach() # Advantage (subtract value prediction).
        else: raise NotImplementedError("Baseline method not recognised.")

    def per_timestep(self, state, action, reward, next_state, done):
        """Operations to perform on each timestep during training."""
        self.ep_rewards.append(reward)

    def per_episode(self):
        """Operations to perform on each episode end during training."""
        policy_loss, value_loss = self.update_on_episode() 
        del self.ep_rewards[:]; del self.ep_log_probs[:]
        if self.V is not None: del self.ep_values[:]
        return {"policy_loss": policy_loss, "value_loss": value_loss}
