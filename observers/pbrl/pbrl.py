from .sampler import Sampler
from .logger import Logger

import os
import torch
from numpy import argwhere
from joblib import load as load_jl, dump


class PbrlObserver:
    """
    xxx
    """
    def __init__(self, P, features, run_names=None, episodes=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.P = P 
        if type(features) == dict: self.feature_names, self.features = list(features.keys()), features
        elif type(features) == list: self.feature_names, self.features = features, None
        self.run_names = run_names if run_names is not None else [] # NOTE: Order crucial to match with episodes
        self.load_episodes(episodes if episodes is not None else [])
        # Reward model, trajectory pair sampler, preference collection interface and logger are all modular
        self.model = self.P["model"]["kind"](self.device, self.feature_names, self.P["model"]) if "model" in self.P else None
        self.sampler = Sampler(self, self.P["sampler"]) if "sampler" in self.P else None
        self.interface = self.P["interface"]["kind"](self, **self.P["interface"]) if "interface" in self.P else None
        self.logger = Logger(self, self.P["logger"]) if "logger" in self.P else None
        self._k = 0 # Preference counter
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
            
    def load_episodes(self, episodes):
        """
        Load a dataset of episodes and initialise data structures.
        """
        self.episodes = episodes
        self.Pr = torch.full((len(episodes), len(episodes)), float("nan"))
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

    def feature_map(self, transitions):
        """
        Map an array of transitions to an array of features.
        """
        if self.features is None: return transitions
        return torch.cat([self.features[f](transitions).reshape(-1,1) for f in self.feature_names], dim=1)

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
            mu, _, std = self.model(self.feature_map(transitions))        
        if "rune_coef" in self.P: return mu + self.P["rune_coef"] * std
        else: return mu

    def fitness(self, trajectory):
        return self.model.fitness(self.feature_map(trajectory))

# ==============================================================================
# METHODS FOR EXECUTING THE LEARNING PROCESS

    def preference_batch(self, batch_size=1, ij_min=0): 
        """
        Sample a batch of trajectory pairs and collect preferences via the interface.
        """
        budget = self.P["feedback_budget"] if "feedback_budget" in self.P else float("inf")
        self.sampler.batch_size, self.sampler.ij_min = batch_size, ij_min
        with self.interface:
            for exit_code, i, j, _ in self.sampler:
                if exit_code == 0:
                    y_ij = self.interface(i, j)
                    if y_ij == "esc": print("=== Feedback exited ==="); break
                    elif y_ij == "skip": print(f"({i}, {j}) skipped"); continue
                    self.log_preference(i, j, y_ij)
                    readout = f"{self.sampler._k} / {batch_size} ({self._k} / {budget}): P({i} > {j}) = {y_ij}"
                    print(readout); self.interface.print("\n"+readout)
                elif exit_code == 1: print("=== Batch complete ==="); break
                elif exit_code == 2: print("=== Fully connected ==="); break

    def log_preference(self, i, j, y_ij):
        assert torch.isnan(self.Pr[i, j]) and torch.isnan(self.Pr[j, i]), f"Already have preference for ({i}, {j})"
        assert 0 <= y_ij <= 1
        self.Pr[i, j] = y_ij
        self.Pr[j, i] = 1 - y_ij
        self._k += 1

    def update(self, history_key):
        """
        Update the reward function to reflect the current preference dataset.
        """
        # Assemble data structures needed for learning
        A, y, i_list, j_list, connected = self.construct_A_and_y()
        print(f"Connected episodes: {len(connected)} / {len(self.episodes)}")
        if len(connected) == 0: print("=== None connected ==="); return
        ep_lengths = [len(self.episodes[i]) for i in connected]
        # Apply feature mapping to all episodes that are connected to the preference graph
        features = self.feature_map(torch.cat([self.episodes[i] for i in connected]))
        # Update the reward function using connected episodes
        logs = self.model.update(history_key, features, ep_lengths, A, i_list, j_list, y)
        # If applicable, relabel the agent's replay memory using the updated reward function
        self.relabel_memory()
        return logs

    def construct_A_and_y(self):
        """
        Construct A and y matrices from the matrix of preference probabilities.
        """
        pairs, y, connected = [], [], set()
        for i, j in argwhere(~torch.isnan(self.Pr)).T: # NOTE: PyTorch v1.10 doesn't have argwhere
            if j < i: 
                i, j = i.item(), j.item()
                pairs.append([i, j]); y.append(self.Pr[i, j]); connected = connected | {i, j}
        y = torch.tensor(y, device=self.device).float()
        connected = sorted(list(connected))
        A = torch.zeros((len(pairs), len(connected)), device=self.device)
        i_list, j_list = [], []
        for l, (i, j) in enumerate(pairs): 
            i_c, j_c = connected.index(i), connected.index(j)
            A[l, i_c], A[l, j_c] = 1, -1
            i_list.append(i_c); j_list.append(j_c)
        return A, y, i_list, j_list, connected

    def relabel_memory(self): pass

# ==============================================================================
# METHODS SPECIFIC TO ONLINE LEARNING

    def per_timestep(self, ep, t, state, action, next_state, reward, done, info, extra):     
        """
        Store transition for current timestep.
        """
        if "discrete_action_map" in self.P: action = self.P["discrete_action_map"][action] 
        self._current_ep.append(list(state) + list(action) + list(next_state))
            
    def per_episode(self, ep): 
        """
        Operations to complete at the end of an episode, which may include storing self._current_ep
        in self.episodes, creating logs, and (if self._online==True), occasionally gathering
        a preference batch and updating the reward function.
        """   
        self._current_ep = torch.tensor(self._current_ep, device=self.device).float() # Convert to tensor once appending finished
        logs = {}
        # Log reward sums
        if self.P["reward_source"] == "model": 
            logs["reward_sum_model"] = self.fitness(self._current_ep)[0].item()
        if self.interface is not None and self.interface.oracle is not None: 
            logs["reward_sum_oracle"] = sum(self.interface.oracle(self._current_ep)).item()
        # Retain episodes for use in reward inference with a specified frequency
        if self._observing and (ep+1) % self.P["observe_freq"] == 0: 
            self.episodes.append(self._current_ep); n = len(self.episodes)
            Pr_old = self.Pr; self.Pr = torch.full((n, n), float("nan")); self.Pr[:-1,:-1] = Pr_old
        if self._online:
            if (ep+1) % self.P["feedback_freq"] == 0 and (ep+1) <= self.P["num_episodes_before_freeze"]:    
                # Calculate batch size.
                # K = self.P["feedback_budget"] 
                # B = self._num_batches 
                # f = self.P["feedback_freq"] / self.P["observe_freq"] # Number of episodes between batches
                # c = self.P["scheduling_coef"] 
                # b = self._batch_num # Current batch number.
                # batch_size = int(round((K / B * (1 - c)) + (K * (f * (2*(b+1) - 1) - 1) / (B * (B*f - 1)) * c)))
                assert self.P["scheduling_coef"] == 0
                K = self.P["feedback_budget"] - self._k # Remaining budget
                B = self._num_batches - self._batch_num # Remaining number of batches
                batch_size = int(round(K / B))
                # Gather preferences and update reward function
                self.preference_batch(batch_size=batch_size, ij_min=self._n_on_prev_batch)
                self._batch_num += 1 
                self._n_on_prev_batch = len(self.episodes)
                logs.update(self.update(history_key=(ep+1)))
            logs["feedback_count"] = self._k
            # Periodically log and save out
            if self.logger is not None and (ep+1) % self.logger.P["freq"] == 0: self.logger(history_key=(ep+1))   
        if self._saving and (ep+1) % self.P["save_freq"] == 0: self.save(history_key=(ep+1))
        self._current_ep = []
        return logs

# ==============================================================================
# SAVING/LOADING

    def save(self, history_key):
        path = f"models/{self.run_names[-1]}"
        if not os.path.exists(path): os.makedirs(path)
        dump({
              "episodes": self.episodes,
              "Pr": self.Pr,
              "model": self.model
        }, f"{path}/{history_key}.pbrl")

def load(fname, P, features):
    """
    Make an instance of PbRLObserver from the information stored by the .save() method.
    """
    dict = load_jl(fname)
    pbrl = PbrlObserver(P, features=features, episodes=dict["episodes"])
    pbrl.Pr = dict["Pr"]
    if dict["model"] is not None:
        assert pbrl.model is None, "New/existing model conflict."
        pbrl.model = dict["model"]
    # pbrl.compute_r_and_var()
    # A, y, _, _, connected = pbrl.construct_A_and_y(pbrl.Pr)
    # ep_fitness_cv = fitness_case_v(A, y, pbrl.P["p_clip"])
    # NOTE: pbrl.episodes contains the featurised representation of each transition.
    # pbrl.episodes = [None for _ in range(pbrl.Pr.shape[0])]
    # for i, ep in zip(pbrl._connected, hr.group_along_dim(dict["tree"].space, "ep")):
    #     assert i == ep[0,0], "pbrl._connected does not match episodes in tree."
    #     pbrl.episodes[i] = ep[:,2:] # Remove ep and reward columns
    print(f"Loaded {fname}")
    return pbrl
