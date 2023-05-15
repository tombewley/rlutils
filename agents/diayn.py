from .sac import SacAgent # DIAYN inherits from SAC.
from ._default_hyperparameters import default_hyperparameters
from ..common.networks import SequentialNetwork
from ..common.utils import col_concat, one_hot, from_numpy

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium.spaces import Box


class DiaynAgent(SacAgent):
    """
    Diversity Is All You Need (DIAYN).
    """
    def __init__(self, env, hyperparameters):
        # Overwrite default hyperparameters for SAC.
        P = default_hyperparameters["sac"]
        for k, v in default_hyperparameters["diayn"].items(): P[k] = v
        for k, v in hyperparameters.items(): P[k] = v
        # Augment the observation space with a one-hot skill vector before initialising SAC.
        P["aug_obs_space"] = [env.observation_space, Box(0., 1., (P["num_skills"],))]
        SacAgent.__init__(self, env, P)
        # Create skill discriminator network, optionally accepting action and/or next state alongside current state.
        input_disc = [env.observation_space]
        if self.P["include_actions"]: input_disc += [env.action_space]
        if self.P["include_next_states"]: input_disc += [env.observation_space]
        self.discriminator = SequentialNetwork(code=self.P["net_disc"], input_space=input_disc, output_size=self.P["num_skills"],
                                               normaliser=self.P["input_normaliser"], lr=self.P["lr_disc"], device=self.device)
        # Skill distribution.
        self.p_z = np.full(self.P["num_skills"], 1.0 / self.P["num_skills"]) # NOTE: Uniform.
        # Tracking variables.
        self.skill = self._sample_skill() # Initialise skill.
        self.ep_losses_discriminator = []
        self.ep_pseudo_return = 0.

    def act(self, state, skill=None, explore=True, do_extra=False):
        """Augment state with one-hot skill vector, then use SAC action selection."""
        if skill is None: skill = self.skill
        state_aug = col_concat(state, one_hot(skill, self.P["num_skills"], self.device))
        action, extra = SacAgent.act(self, state_aug, explore, do_extra)
        if do_extra: extra["skill"] = skill
        return action, extra

    def update_on_batch(self):
        """Use random batches from the replay memory to update the discriminator, pi and Q network parameters."""
        states, actions, _, _, next_states = self.memory.sample(self.P["batch_size"], keep_terminal_next=True)
        if states is None: return
        features, zs = torch.split(states, [self.env.observation_space.shape[0], self.P["num_skills"]], dim=1)
        if self.P["include_actions"]: features = col_concat(features, actions)
        if self.P["include_next_states"]: features = col_concat(features, next_states[:, :-self.P["num_skills"]])
        # Update discriminator to minimise cross-entropy loss against skills.
        loss = F.cross_entropy(self.discriminator(features), zs.argmax(1))   
        self.discriminator.optimise(loss)
        self.ep_losses_discriminator.append(loss.item()) # Keeping separate prevents confusion of SAC methods.
        # Execute the SAC update function for pi and Q and return losses.
        return SacAgent.update_on_batch(self)

    def per_timestep(self, state, action, _, next_state, done, skill=None):
        """Operations to perform on each timestep during training."""
        # Augment state and next state with one-hot skill vector and compute diversity reward.
        if skill is None: skill = self.skill
        z = one_hot(skill, self.P["num_skills"], self.device)
        features = state
        if self.P["include_actions"]: features = col_concat(features, from_numpy(action, device=self.device))
        if self.P["include_next_states"]: features = col_concat(features, next_state)
        pseudo_reward = self._pseudo_reward(features, skill)
        self.ep_pseudo_return += pseudo_reward
        SacAgent.per_timestep(self, col_concat(state, z), action, pseudo_reward, col_concat(next_state, z), done)

    def per_episode(self):
        """Operations to perform on each episode end during training."""
        logs = SacAgent.per_episode(self)
        logs["discriminator_loss"] = np.mean(self.ep_losses_discriminator) if self.ep_losses_discriminator else 0.
        logs["pseudo_return"] = self.ep_pseudo_return
        del self.ep_losses_discriminator[:]; self.ep_pseudo_return = 0.
        self.skill = self._sample_skill() # Resample skill for the next episode.
        return logs

    def _sample_skill(self):
        """Sample skill according to probabilities in self.p_z.""" 
        return np.random.choice(self.P["num_skills"], p=self.p_z)

    def _pseudo_reward(self, features, skill):
        """Construct diversity-promoting pseudo-reward using skill discriminator network."""
        # Log conditional probability of skill according to discriminator.
        reward = - F.cross_entropy(self.discriminator(features), torch.tensor([skill], device=self.device)).item()
        if self.P["log_p_z_in_reward"]:
            # Log prior probability of skill. Subtracting this means rewards are non-negative as long as the discriminator does better than chance.
            reward -= np.log(self.p_z[skill])
        return reward
