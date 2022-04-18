from .graph import PreferenceGraph
from .sampler import Sampler
from .explainer import Explainer
from .interactions import preference_batch, update_model

import os
import torch


class PbrlObserver:
    """
    xxx
    """
    def __init__(self, P, run_names=None, episodes=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.P = P
        self.run_names = run_names if run_names is not None else [] # NOTE: Order crucial to match with episodes
        # Preference graph, reward model, trajectory pair sampler, preference collection interface and explainer are all modular
        self.graph = PreferenceGraph(self.device, episodes)
        self.model = self.P["model"]["class"](self.P["model"]) if "model" in self.P else None
        self.sampler = Sampler(self, self.P["sampler"]) if "sampler" in self.P else None
        self.interface = self.P["interface"]["class"](self, self.P["interface"]) if "interface" in self.P else None
        self.explainer = Explainer(self, self.P["explainer"] if "explainer" in self.P else {})
        self._observing = "observe_freq" in self.P and self.P["observe_freq"] > 0
        self._saving = "save_freq" in self.P and self.P["save_freq"] > 0
        self._online = "feedback_freq" in self.P and self.P["feedback_freq"] > 0
        if self._online:
            assert self.model is not None and self.sampler is not None and self.interface is not None
            assert self._observing
            assert self.P["feedback_freq"] % self.P["observe_freq"] == 0    
            b = self.P["num_episodes_before_freeze"] / self.P["feedback_freq"]
            assert b % 1 == 0
            self._num_batches = int(b)
            self._batch_num = 0
            self._n_on_prev_batch = 0
        self._current_ep = []
    
    def link(self, agent):
        """
        Link the reward model to the replay memory of an off-policy RL agent.
        """
        assert len(agent.memory) == 0, "Agent must be at the start of learning."
        assert agent.device == self.device
        agent.memory.__init__(agent.memory.capacity, reward=self.reward, relabel_mode="eager")
        if not agent.memory.lazy_reward: self.relabel_memory = agent.memory.relabel

# ==============================================================================
# PREDICTION METHODS

    def reward(self, states, actions, next_states, return_params=False):
        """
        Reward function, defined over individual transitions (s,a,s').
        """
        assert self.P["reward_source"] != "extrinsic", "This shouldn't have been called. Unwanted call to pbrl.link(agent)?"
        if "discrete_action_map" in self.P: actions = [self.P["discrete_action_map"][a] for a in actions] 
        transitions = torch.cat([states, actions, next_states], dim=-1)
        if self.P["reward_source"] == "oracle":
            assert not return_params, "Oracle doesn't use normal distribution parameters"
            return self.interface.oracle(transitions)
        else:
            mu, _, std = self.model(transitions)
        if "rune_coef" in self.P: return mu + self.P["rune_coef"] * std
        else: return mu

# ==============================================================================
# METHODS SPECIFIC TO ONLINE LEARNING

    def per_timestep(self, ep_num, t, state, action, next_state, reward, done, info, extra):
        """
        Store transition for current timestep.
        """
        if "discrete_action_map" in self.P: action = self.P["discrete_action_map"][action]
        self._current_ep.append(list(state) + list(action) + list(next_state)) # TODO: Keep (s,a,s') separate when store in graph
            
    def per_episode(self, ep_num):
        """
        Operations to complete at the end of an episode, which may include adding self._current_ep
        to the preference graph, creating logs, and (if self._online==True), occasionally gathering
        a preference batch and updating the reward model.
        """   
        self._current_ep = torch.tensor(self._current_ep, device=self.device).float() # Convert to tensor once appending finished
        logs = {}
        # Log reward sums
        if self.P["reward_source"] == "model": 
            logs["reward_sum_model"] = self.model.fitness(self._current_ep)[0].item()
        if self.interface is not None and self.interface.oracle is not None: 
            logs["reward_sum_oracle"] = sum(self.interface.oracle(self._current_ep)).item()
        # Add episodes to the preference graph with a specified frequency
        if self._observing and (ep_num+1) % self.P["observe_freq"] == 0:
            self.graph.add_episode(run_name=self.run_names[-1], ep_num=ep_num, transitions=self._current_ep)
        if self._online:
            if (ep_num+1) % self.P["feedback_freq"] == 0 and (ep_num+1) <= self.P["num_episodes_before_freeze"]:
                # Calculate batch size.
                if self.P["scheduling_coef"] > 0: # TODO: Make adaptive to remaining budget
                    assert self.sampler.P["recency_constraint"]
                    K = self.P["feedback_budget"]
                    B = self._num_batches
                    f = self.P["feedback_freq"] / self.P["observe_freq"] # Number of episodes between batches
                    c = self.P["scheduling_coef"]
                    b = self._batch_num # Current batch number.
                    batch_size = int(round((K / B * (1 - c)) + (K * (f * (2*(b+1) - 1) - 1) / (B * (B*f - 1)) * c)))
                else:
                    K = self.P["feedback_budget"] - len(self.graph.edges) # Remaining budget
                    B = self._num_batches - self._batch_num # Remaining number of batches
                    batch_size = int(round(K / B))
                # Gather preferences and update reward model
                logs.update(preference_batch(
                    sampler=self.sampler,
                    interface=self.interface,
                    graph=self.graph,
                    batch_size=batch_size,
                    ij_min=self._n_on_prev_batch,
                    history_key=(ep_num+1),
                    budget=self.P["feedback_budget"]
                ))
                logs.update(update_model(
                    graph=self.graph,
                    model=self.model,
                    history_key=(ep_num+1)
                ))
                self.relabel_memory() # If applicable, relabel the agent's replay memory using the updated reward
                self._batch_num += 1 
                self._n_on_prev_batch = len(self.graph)
            # Periodically log and save out
            if self.explainer.P and (ep_num+1) % self.explainer.P["freq"] == 0: self.explainer(history_key=(ep_num+1))
        if self._saving and (ep_num+1) % self.P["save_freq"] == 0: self.save(history_key=(ep_num+1))
        self._current_ep = []
        return logs

    def relabel_memory(self): pass

# ==============================================================================
# SAVING/LOADING

    def save(self, history_key):
        path = f"models/{self.run_names[-1]}"
        if not os.path.exists(path): os.makedirs(path)
        torch.save({
            "graph": self.graph,
            "model": self.model
        }, f"{path}/{history_key}.pbrl")

def load(fname, P):
    """
    Make an instance of PbrlObserver from the information stored by the .save() method.
    """
    dict = torch.load(fname, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    pbrl = PbrlObserver(P)
    pbrl.graph = dict["graph"]
    pbrl.graph.device = pbrl.device
    if dict["model"] is not None:
        assert pbrl.model is None, "New/existing model conflict."
        pbrl.model = dict["model"]
        pbrl.model.device = pbrl.device
    print(f"Loaded {fname}")
    return pbrl
