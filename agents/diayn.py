from .sac import SacAgent # DIAYN inherits from SAC.
from ._default_hyperparameters import default_hyperparameters
from ..common.networks import SequentialNetwork
from ..common.utils import col_concat, one_hot, from_numpy

import torch
import torch.nn.functional as F
from numpy import mean
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
        self._discriminator = SequentialNetwork(code=self.P["net_disc"], input_space=input_disc, output_size=self.P["num_skills"],
                                               normaliser=self.P["input_normaliser"], lr=self.P["lr_disc"], device=self.device)
        # Skill distribution.
        self.p_z = torch.full((self.P["num_skills"],), 1.0 / self.P["num_skills"], device=self.device) # NOTE: Uniform.
        # Tracking variables.
        self.skill = self.sample_skill() # Initialise skill.
        self.ep_losses_discriminator = []
        self.ep_pseudo_return = 0.

    def act(self, state, skill=None, explore=True, do_extra=False):
        """Augment state with one-hot skill vector, then use SAC action selection."""
        if skill is None: skill = self.skill
        state_aug = col_concat(state, one_hot(skill.item(), self.P["num_skills"], self.device))
        action, extra = SacAgent.act(self, state_aug, explore, do_extra)
        if do_extra: extra["skill"] = skill
        return action, extra

    def update_on_batch(self):
        """Use random batches from the replay memory to update the discriminator, pi and Q network parameters."""
        states_aug, actions, _, _, next_states_aug = self.memory.sample(self.P["batch_size"], keep_terminal_next=True)
        if states_aug is None: return
        states, zs = torch.split(states_aug, [self.env.observation_space.shape[0], self.P["num_skills"]], dim=1)
        # Update discriminator to minimise cross-entropy loss against skills.
        loss = F.cross_entropy(self.discriminator(states, actions, next_states_aug[:, :-self.P["num_skills"]]), zs.argmax(1))
        self._discriminator.optimise(loss)
        self.ep_losses_discriminator.append(loss.item()) # Keeping separate prevents confusion of SAC methods.
        # Sample a new batch, compute latest pseudo-rewards, pass to the SAC update function and return losses.
        states_aug, actions, _, nonterminal_mask, next_states_aug = self.memory.sample(self.P["batch_size"], keep_terminal_next=True)
        states, zs = torch.split(states_aug, [self.env.observation_space.shape[0], self.P["num_skills"]], dim=1)
        pseudo_rewards = self.pseudo_reward(states, actions, next_states_aug[:, :-self.P["num_skills"]], zs.argmax(1))
        return SacAgent.update_on_batch(self, (states_aug, actions, pseudo_rewards, nonterminal_mask, next_states_aug[nonterminal_mask]))

    def per_timestep(self, state, action, _, next_state, done):
        """Operations to perform on each timestep during training."""
        z = one_hot(self.skill.item(), self.P["num_skills"], self.device)
        self.ep_pseudo_return += self.pseudo_reward(state, from_numpy(action, device=self.device), next_state, self.skill)
        # Augment state and next state with one-hot skill vector, but *don't* store diversity reward (would be out-of-date when sampled).
        SacAgent.per_timestep(self, col_concat(state, z), action, 0., col_concat(next_state, z), done)

    def per_episode(self):
        """Operations to perform on each episode end during training."""
        logs = SacAgent.per_episode(self)
        logs["discriminator_loss"] = mean(self.ep_losses_discriminator) if self.ep_losses_discriminator else 0.
        logs["pseudo_return"] = self.ep_pseudo_return
        del self.ep_losses_discriminator[:]; self.ep_pseudo_return = 0.
        self.skill = self.sample_skill() # Resample skill for the next episode.
        return logs

    def sample_skill(self):
        """Sample skill according to probabilities in self.p_z."""
        return torch.multinomial(self.p_z, 1)

    def discriminator(self, states, actions, next_states):
        """Pass transitions to skill dicriminator network."""
        return self._discriminator(torch.cat([states] + ([actions] if self.P["include_actions"] else [])
                                             + ([next_states] if self.P["include_next_states"] else []), dim=-1))

    def pseudo_reward(self, states, actions, next_states, skills):
        """Construct diversity-promoting pseudo-reward using skill discriminator network."""
        # Log conditional probability of skill according to discriminator.
        with torch.no_grad():
            reward = - F.cross_entropy(self.discriminator(states, actions, next_states), skills, reduction="none")
        if self.P["log_p_z_in_reward"]:
            # Log prior probability of skill. Subtracting this means rewards are non-negative as long as the discriminator does better than chance.
            reward -= torch.log(self.p_z[skills])
        return reward
