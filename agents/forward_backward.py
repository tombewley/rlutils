from ._generic import Agent
from ..common.networks import SequentialNetwork
from ..common.memory import ReplayMemory
from ..common.exploration import EpsilonGreedy
from ..common.featuriser import Featuriser

import torch
import torch.nn.functional as F
from torch.distributions.cauchy import Cauchy
from numpy import mean
from gymnasium.spaces.discrete import Discrete
from gymnasium.spaces.box import Box


class ForwardBackwardAgent(Agent):
    """
    Agent for learning a forward-backward (FB) representation of an MDP from reward-free interactions. From:
        Touati, Ahmed, and Yann Ollivier.
        "Learning one representation to optimize all rewards."
        Advances in Neural Information Processing Systems 34 (2021): 13-23.
    NOTE: Currently requires a discrete action space.
    TODO: Consult updates in "Does Zero-Shot Reinforcement Learning Exist?" (Appendix F).
    """
    def __init__(self, env, hyperparameters):
        Agent.__init__(self, env, hyperparameters)
        self.discrete_states = type(self.env.observation_space) == Discrete
        # One-hot representation of discrete actions and (if applicable) states.
        n_a = self.env.action_space.n
        action_space = Box(shape=(n_a,), low=0., high=1.)
        self.one_hot = torch.eye(n_a, device=self.device)
        if self.discrete_states:
            n_s = self.env.observation_space.n
            state_space = Box(shape=(n_s,), low=0., high=1.)
            self.one_hot_s = torch.eye(n_s, device=self.device)
        else: state_space = self.env.observation_space
        # Create forward (F) network and its target.
        embedding_space = Box(shape=(self.P["embed_dim"],), low=-float("inf"), high=float("inf"))
        input_F = [state_space, action_space, embedding_space]
        self._F        = SequentialNetwork(code=self.P["net_FB"], input_space=input_F,
                                           output_size=self.P["embed_dim"], normaliser=self.P["input_normaliser"],
                                           device=self.device)
        self._F_target = SequentialNetwork(code=self.P["net_FB"], input_space=input_F,
                                           output_size=self.P["embed_dim"], normaliser=self.P["input_normaliser"],
                                           eval_only=True, device=self.device)
        self._F_target.load_state_dict(self._F.state_dict())  # Clone.
        # Create backward (B) network and its target.
        # NOTE: B can use a featurised representation of (s, a, s').
        if "featuriser" in self.P: self.featuriser = self.P["featuriser"]
        else:
            n_s = state_space.shape[0]
            self.featuriser = Featuriser({"feature_names":
                [f"s{i}" for i in range(n_s)] + [f"a{i}" for i in range(n_a)] + [f"ns{i}" for i in range(n_s)]})
        input_B = [Box(shape=(len(self.featuriser.names), ), low=-float("inf"), high=float("inf"))]
        self._B        = SequentialNetwork(code=self.P["net_FB"], input_space=input_B,
                                           output_size=self.P["embed_dim"], normaliser=self.P["input_normaliser"],
                                           device=self.device)
        self._B_target = SequentialNetwork(code=self.P["net_FB"], input_space=input_B,
                                           output_size=self.P["embed_dim"], normaliser=self.P["input_normaliser"],
                                           eval_only=True, device=self.device)
        self._B_target.load_state_dict(self._B.state_dict())  # Clone.
        # Create Adam optimser to jointly handle the parameters of both networks.
        self.optimiser = torch.optim.Adam(list(self._F.parameters()) + list(self._B.parameters()), lr=self.P["lr_FB"])
        # Create replay memory.
        self.memory = ReplayMemory(self.P["replay_capacity"])
        # Initialise epsilon-greedy exploration.
        self.exploration = EpsilonGreedy(self.P["epsilon"], 0, 1) # NOTE: Epsilon doesn't decay.
        # Create commonly-reused constants and class instances.
        self.max_policy_entropy = torch.tensor(n_a, device=self.device).log()
        self.cauchy = Cauchy(torch.tensor([0.0], device=self.device), torch.tensor([0.5], device=self.device))
        self.softmax = torch.nn.Softmax(dim=1)
        # Tracking variables.
        self.z = self._sample_z()  # Initialise preference vector.
        self.ep_metrics = []

    def act(self, state, explore=True, do_extra=False):
        """Epsilon-greedy action selection."""
        with torch.no_grad():
            return self.exploration(self.Q(state, self.z).squeeze(), explore, do_extra)

    def update_on_batch(self):
        """Use pairs of random batches from the replay memory to update the F and B network parameters."""
        # Sample two batches of transitions, one "current" and one "future".
        states, actions, _, nonterminal_mask, next_states = self.memory.sample(self.P["batch_size"])
        if states is None: return
        # NOTE: Only use transitions where the next state is nonterminal.
        b = len(next_states)
        states, actions = states[nonterminal_mask], actions[nonterminal_mask]
        states_f, actions_f, _, _, next_states_f = self.memory.sample(b, keep_terminal_next=True)
        # Sample a preference vector z for each transition.
        z = self._sample_z(num=b)
        # Predict successor measure M(s, a, s_f, a_f, s'_f, z) for all pairwise combinations from the two batches.
        M_values = self.M_pairwise(states, actions, states_f, actions_f, next_states_f, z)
        # Predict successor measure for immediate transition following each current state-action pair.
        M_immediate = self.M(states, actions, states, actions, next_states, z)
        # Use target networks to compute M(s', a', s_f, a_f, s'_f, z) for all pairwise combinations.
        # Rather than taking the greedy a', use a weighted average over a softmax policy.
        next_action_probs = self.softmax(self.Q(next_states, z, target=True) / self.P["softmax_tau"])
        # For monitoring, compute mean entropy of next_action_probs relative to the maximum possible.
        next_action_entropy = -(next_action_probs * next_action_probs.log()).nansum(dim=1).mean() / self.max_policy_entropy
        next_M_values = torch.zeros_like(M_values)
        for i, p_a in enumerate(next_action_probs.T):
            next_actions = torch.full(fill_value=i, size=(b,), device=self.device, dtype=torch.int64)
            next_M_values += p_a * self.M_pairwise(next_states, next_actions, states_f, actions_f, next_states_f, z,
                                                   target=True).detach()
        # Compute MSE of modified TD error (line 19 of Algorithm 1).
        td_loss = (F.mse_loss(M_values, self.P["gamma"] * next_M_values) / 2) - M_immediate.mean()
        # Compute orthonormality regularisation loss for B.
        B = self.B(states, actions, next_states)
        B_f = self.B(states_f, actions_f, next_states_f)
        B_exp = B.unsqueeze(1).expand(b, b, self.P["embed_dim"])
        B_f_exp = B_f.unsqueeze(0).expand(b, b, self.P["embed_dim"])
        BT_B_f = (B_exp * B_f_exp.detach()).sum(dim=-1)
        reg_loss = (BT_B_f * BT_B_f.detach()).mean() - (B * B.detach()).sum(dim=-1).mean()
        # reg_loss = ((B * B.detach()).sum(dim=-1) * BT_B_f.detach()).mean() - BT_B_f.mean()  # NOTE: Used on GitHub.
        # Update both networks.
        self.optimiser.zero_grad()
        (td_loss + self.P["lambda"] * reg_loss).backward()
        for param in self._F.parameters(): param.grad.data.clamp_(-1, 1)
        for param in self._B.parameters(): param.grad.data.clamp_(-1, 1)
        self.optimiser.step()
        # Perform soft (Polyak) updates on targets.
        for net, target in ((self._F, self._F_target), (self._B, self._B_target)): target.polyak(net, tau=self.P["tau"])
        return td_loss.item(), reg_loss.item(), next_action_entropy.item()

    def per_timestep(self, state, action, reward, next_state, done):
        """Operations to perform on each timestep during training."""
        self.memory.add(state, action, reward, next_state, done)
        metrics = self.update_on_batch()
        if metrics: self.ep_metrics.append(metrics)

    def per_episode(self):
        """Operations to perform on each episode end during training."""
        mean_td_loss, mean_reg_loss, mean_entropy = mean(self.ep_metrics, axis=0) if self.ep_metrics else (0., 0., 0.)
        del self.ep_metrics[:]
        self.z = self._sample_z()  # Resample preference vector for the next episode.
        return {"td_loss": mean_td_loss, "reg_loss": mean_reg_loss, "next_action_entropy": mean_entropy}

    def Q(self, states, z, target=False):
        """Predict state-action value Q(s, a) = F(s, a, z)ᵀz."""
        b, n_a = states.shape[0], self.one_hot.shape[0]
        states = states.unsqueeze(1).expand(b, n_a, *states.shape[1:])
        actions = torch.arange(n_a).unsqueeze(0).expand(b, n_a)
        z = z.unsqueeze(1).expand(b, n_a, self.P["embed_dim"])
        return (self.F(states, actions, z, target=target) * z).sum(dim=-1)

    def M_pairwise(self, states, actions, states_f, actions_f, next_states_f, z, target=False):
        """Predict pairwise successor measures using self.M."""
        return torch.cat([self.M(states[i:i+1], actions[i:i+1], states_f, actions_f, next_states_f, z[i:i+1]
                                 ).unsqueeze(0) for i in range(states.shape[0])])

    def M(self, states, actions, states_f, actions_f, next_states_f, z, target=False):
        """Predict successor measure M(s, a, s_f, a_f, s'_f, z) = F(s, a, z)ᵀB(s_f, a_f, s'_f)."""
        return (self.F(states, actions, z, target=target)
              * self.B(states_f, actions_f, next_states_f, target=target)).sum(dim=-1)

    def F(self, states, actions, z, target=False):
        """Predict successor features for state-action pairs."""
        if self.discrete_states: states = self.one_hot_s[states]
        # NOTE: GitHub implementation seems to normalise z.
        return (self._F_target if target else self._F)(torch.cat((states, self.one_hot[actions], z), dim=-1))

    def B(self, states, actions, next_states, target=False):
        """Predict predecessor features for transitions."""
        if self.discrete_states: states, next_states = self.one_hot_s[states], self.one_hot_s[next_states]
        return (self._B_target if target else self._B)(self.featuriser(states, self.one_hot[actions], next_states))

    def _sample_z(self, num=1, eps=1e-10):
        """Sample preference vector z on a hypersphere surface, optionally rescaled by a Cauchy variable."""
        z = torch.normal(mean=0., std=1., size=(num, self.P["embed_dim"]), device=self.device)
        z *= (self.P["embed_dim"] ** 0.5) / (torch.norm(z, dim=-1, keepdim=True) + eps)
        if self.P["cauchy_z"]: z *= self.cauchy.sample((num,))
        return z
