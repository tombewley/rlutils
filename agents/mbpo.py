from .sac import SacAgent # MBPO inherits from SAC.
from ._default_hyperparameters import default_hyperparameters
from ..common.dynamics import DynamicsModel

import torch
from numpy import mean
from random import sample as random_sample


class MbpoAgent(SacAgent):
    """
    Model-based policy optimisaton. From:
        Janner, Michael, Justin Fu, Marvin Zhang, and Sergey Levine. 
        "When to trust your model: Model-based policy optimization."
        Advances in Neural Information Processing Systems 32 (2019).
    """
    def __init__(self, env, hyperparameters):
        assert "reward" in hyperparameters, f"{type(self).__name__} requires a reward function."
        # Overwrite default hyperparameters for SAC.
        P = default_hyperparameters["sac"]
        for k, v in default_hyperparameters["mbpo"].items(): P[k] = v
        for k, v in hyperparameters.items(): P[k] = v
        SacAgent.__init__(self, env, P)
        # Create dynamics model.
        self.P["rollout"]["num_particles"] = 1
        self.model = DynamicsModel(code=self.P["net_model"], observation_space=self.env.observation_space, action_space=self.env.action_space,
                                   reward_function=self.P["reward"], probabilistic=self.P["probabilistic"], lr=self.P["lr_model"],
                                   ensemble_size=self.P["ensemble_size"], rollout_params=self.P["rollout"], device=self.device)
        # Create rollout memory, which behaves similarly to common.ReplayMemory but with a capacity that varies based on model.horizon.
        self.rollouts = RolloutMemory(self.P["retained_updates"])
        # Tracking variables.
        self.random_mode = True
        self.ep_losses_model = []

    def act(self, state, explore=True, do_extra=False):
        """Either random or SAC action selection."""
        with torch.no_grad():
            extra = {}
            if self.random_mode: action = self.env.action_space.sample()
            else: action, extra = SacAgent.act(self, state, explore, do_extra)
            if do_extra: 
                action_torch = torch.tensor(action, device=self.device).unsqueeze(0)
                extra["next_state_pred"] = self.model.predict(state, action_torch)[0].cpu().numpy()
            return action, extra

    def update_on_batch(self):
        """Use random batches from the replay memory to update the model. Then sample a batch of "seed" states,
        use the model to generate synthetic transitions to add to rollouts."""
        if self.total_t % self.P["model_freq"] == 0:
            # Optimise each network in the model ensemble on an independent batch.
            for i in range(self.P["ensemble_size"]):
                states, actions, _, _, next_states = self.memory.sample(self.P["batch_size_model"], keep_terminal_next=True)
                if states is None: return 
                self.ep_losses_model.append(self.model.update_on_batch(states, actions, next_states, i))
            # Periodically increase the model rollout horizon.
            self.model.num_updates += 1; self.model._update_horizon()
            with torch.no_grad():
                # Sample a batch of "seed" states from which to begin model rollouts.
                states_init, _, _, _, _ = self.memory.sample(self.P["rollouts_per_update"])
                # Simulate forward dynamics from states_init using randomly-sampled members of the ensemble,
                # consulting pi for action selection. Save the resultant rollouts.
                self.rollouts.add(*self.model.rollout(states_init, policy=self.pi_no_logprob, ensemble_index="ts1"))
        # Uses batches sampled from model rollouts to update pi/Q.
        # Can perform more updates than normal here; use of model-generated data reduces overfitting risk.
        for _ in range(self.P["policy_updates_per_timestep"]):
            losses = SacAgent.update_on_batch(self, self.rollouts.sample(self.P["batch_size"]))
        return losses # NOTE: Just returning loss from the last batch.

    def per_timestep(self, state, action, reward, next_state, done):
        """Operations to perform on each timestep during training."""
        if self.random_mode and self.total_t >= self.P["num_random_steps"]: 
            self.random_mode = False
            print("Random data collection complete.")
        SacAgent.per_timestep(self, state, action, reward, next_state, done, suppress_update=self.random_mode)

    def per_episode(self):
        """Operations to perform on each episode end during training."""
        logs = SacAgent.per_episode(self)
        logs["model_loss"] = mean(self.ep_losses_model) if self.ep_losses_model else 0.
        logs["horizon"] = self.model.horizon
        logs["random_mode"] = int(self.random_mode)
        del self.ep_losses_model[:]
        return logs

    def pi_no_logprob(self, states):
        actions, _ = self.pi(states)
        return actions


class RolloutMemory:
    def __init__(self, capacity): self.capacity = int(capacity); self.clear()

    def __len__(self): return sum(states.shape[0] for states,_,_,_,_ in self.memory)

    def clear(self): self.memory = []; self.position = 0

    def add(self, states, actions, rewards):
        if len(self.memory) < self.capacity: self.memory.append(None) 
        self.memory[self.position] = (
            torch.flatten(states[:,:-1], end_dim=-2),
            torch.flatten(actions, end_dim=-2),
            torch.flatten(rewards),
            torch.ones(torch.numel(rewards), dtype=bool), # NOTE: nonterminal_mask is all True.
            torch.flatten(states[:,1:], end_dim=-2)
        )
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self) < batch_size: return None, None, None, None, None
        indices = random_sample({(i,j) for i,(states,_,_,_,_) in enumerate(self.memory) 
                                       for j in range(states.shape[0])}, batch_size)
        return (torch.cat([self.memory[i][0][j].unsqueeze(0) for i, j in indices], dim=0), # states
                torch.cat([self.memory[i][1][j].unsqueeze(0) for i, j in indices], dim=0), # actions
                torch.cat([self.memory[i][2][j].unsqueeze(0) for i, j in indices], dim=0), # rewards
                torch.cat([self.memory[i][3][j].unsqueeze(0) for i, j in indices], dim=0), # nonterminal_mask
                torch.cat([self.memory[i][4][j].unsqueeze(0) for i, j in indices], dim=0)) # next_states
