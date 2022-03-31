from ._generic import Agent
from ..common.networks import SequentialNetwork

import numpy as np
import torch
from torch.distributions import Categorical
import torch.nn.functional as F


class ActorCriticAgent(Agent):
    """
    Basic actor-critic for discrete action spaces.
    Differs from ReinforceAgent with baseline="adv" in that one-step TD estimates
    are used in place of MC returns. This allows an update to be made every timestep.
    """
    def __init__(self, env, hyperparameters):
        Agent.__init__(self, env, hyperparameters)
        self.eps = np.finfo(np.float32).eps.item() # Small float used to prevent div/0 errors.
        # Create pi and V networks.
        self.pi = SequentialNetwork(code=self.P["net_pi"], input_space=[self.env.observation_space], output_size=self.env.action_space.n,
                                    normaliser=self.P["input_normaliser"], lr=self.P["lr_pi"], device=self.device)
        self.V = SequentialNetwork(code=self.P["net_V"], input_space=[self.env.observation_space], output_size=1,
                                    normaliser=self.P["input_normaliser"], lr=self.P["lr_V"], device=self.device)
        # Tracking variables.
        self.last_log_prob, self.last_value = None, None
        self.ep_losses = []

    def act(self, state, explore=True, do_extra=False):
        """Probabilistic action selection *without* torch.no_grad() to allow backprop later."""
        action_probs = self.pi(state)
        dist = Categorical(action_probs) 
        action = dist.sample()
        self.last_log_prob, self.last_value = dist.log_prob(action).squeeze(), self.V(state).squeeze()
        return action.item(), {"pi": action_probs.squeeze().cpu().detach().numpy(), "V": self.last_value.item()} if do_extra else {}

    def update_on_transition(self, next_state, reward, done):
        """Use the latest transition to update the policy and value network parameters."""
        # Calculate TD target for value function.
        td_target = reward + (self.P["gamma"] * \
                             (torch.tensor(0., device=self.device) if done else self.V(next_state).squeeze()))
        # Update value in the direction of TD error using MSE loss (NOTE: seems to outperform Huber on CartPole!)
        value_loss = F.mse_loss(self.last_value, td_target)
        # value_loss = F.smooth_l1_loss(self.last_value, td_target)
        self.V.optimise(value_loss)
        # Update policy in the direction of log_prob(a) * TD error.
        policy_loss = -self.last_log_prob * (td_target - self.last_value).detach()
        self.pi.optimise(policy_loss)
        return policy_loss.item(), value_loss.item()

    def per_timestep(self, state, action, reward, next_state, done):
        """Operations to perform on each timestep during training."""
        self.ep_losses.append(self.update_on_transition(next_state, reward, done))
        self.last_log_prob, last_value = None, None

    def per_episode(self):
        """Operations to perform on each episode end during training."""
        policy_loss, value_loss = np.mean(self.ep_losses, axis=0)
        del self.ep_losses[:]
        return {"policy_loss": policy_loss, "value_loss": value_loss}
