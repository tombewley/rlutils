from .graph import PreferenceGraph
from .featuriser import Featuriser
from .sampler import Sampler
from .explainer import Explainer

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
        # Preference graph, featuriser, reward model, trajectory pair sampler, preference collection interface and explainer are all modular
        self.graph = PreferenceGraph(self.device, episodes)
        self.featuriser = Featuriser(self.P["featuriser"]) if "featuriser" in self.P else None
        self.model = self.P["model"]["class"](self.device, self.featuriser.names, self.P["model"]) if "model" in self.P else None
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
        if self.P["reward_source"] == "oracle": # NOTE: Oracle defined over raw transitions rather than features
            assert not return_params, "Oracle doesn't use normal distribution parameters"
            return self.interface.oracle(transitions)
        else:
            mu, _, std = self.model(self.featuriser(transitions))
        if "rune_coef" in self.P: return mu + self.P["rune_coef"] * std
        else: return mu

    def fitness(self, trajectory):
        return self.model.fitness(self.featuriser(trajectory))

# ==============================================================================
# METHODS FOR EXECUTING THE LEARNING PROCESS

    def preference_batch(self, history_key, batch_size=1, ij_min=0):
        """
        Sample a batch of trajectory pairs, collect preferences via the interface, and add them to the graph.
        """
        budget = self.P["feedback_budget"] if "feedback_budget" in self.P else float("inf")
        self.sampler.batch_size, self.sampler.ij_min = batch_size, ij_min
        with self.interface:
            for exit_code, i, j, _ in self.sampler:
                if exit_code == 0:
                    preference = self.interface(i, j)
                    if preference == "esc": print("=== Feedback exited ==="); break
                    elif preference == "skip": print(f"({i}, {j}) skipped"); continue
                    self.graph.add_preference(history_key, i, j, preference)
                    readout = f"{self.sampler._k} / {batch_size} ({len(self.graph.edges)} / {budget}): P({i} > {j}) = {preference}"
                    print(readout); self.interface.print("\n"+readout)
                elif exit_code == 1: print("=== Batch complete ==="); break
                elif exit_code == 2: print("=== Fully connected ==="); break

    def update(self, history_key):
        """
        Update the reward function using the provided preference graph.
        """
        # Assemble data structures needed for learning
        A, y, i_list, j_list, connected = self.graph.construct_A_and_y()
        print(f"Connected episodes: {len(connected)} / {len(self.graph)}")
        if len(connected) == 0: print("=== None connected ==="); return {}
        # Get lengths and apply feature mapping to all episodes that are connected to the preference graph
        connected_ep_transitions = [self.graph.nodes[i]["transitions"] for i in connected]
        ep_lengths = [len(tr) for tr in connected_ep_transitions]
        features = self.featuriser(torch.cat(connected_ep_transitions)) # TODO: Don't concatenate here
        # Update the reward function using connected episodes
        logs = self.model.update(history_key, features, ep_lengths, A, i_list, j_list, y)
        return logs

# ==============================================================================
# METHODS SPECIFIC TO ONLINE LEARNING

    def per_timestep(self, ep_num, t, state, action, next_state, reward, done, info, extra):
        """
        Store transition for current timestep.
        """
        if "discrete_action_map" in self.P: action = self.P["discrete_action_map"][action]
        self._current_ep.append(list(state) + list(action) + list(next_state))
            
    def per_episode(self, ep_num):
        """
        Operations to complete at the end of an episode, which may include adding self._current_ep
        to the preference graph, creating logs, and (if self._online==True), occasionally gathering
        a preference batch and updating the reward function.
        """   
        self._current_ep = torch.tensor(self._current_ep, device=self.device).float() # Convert to tensor once appending finished
        logs = {}
        # Log reward sums
        if self.P["reward_source"] == "model": 
            logs["reward_sum_model"] = self.fitness(self._current_ep)[0].item()
        if self.interface is not None and self.interface.oracle is not None: 
            logs["reward_sum_oracle"] = sum(self.interface.oracle(self._current_ep)).item()
        # Add episodes to the preference graph with a specified frequency
        if self._observing and (ep_num+1) % self.P["observe_freq"] == 0:
            self.graph.add_episode(ep_num, transitions=self._current_ep)
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
                # Gather preferences and update reward function
                self.preference_batch(history_key=(ep_num+1), batch_size=batch_size, ij_min=self._n_on_prev_batch)
                self._batch_num += 1 
                self._n_on_prev_batch = len(self.graph)
                logs.update(self.update(history_key=(ep_num+1)))
                # If applicable, relabel the agent's replay memory using the updated reward function
                self.relabel_memory()
            logs["feedback_count"] = len(self.graph.edges)
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
    if dict["model"] is not None:
        assert pbrl.model is None, "New/existing model conflict."
        pbrl.model = dict["model"]
    print(f"Loaded {fname}")
    return pbrl
