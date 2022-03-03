from .logging import log

import os
import torch
import numpy as np
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
        self.model = self.P["model"]["kind"](self.device, self.feature_names, self.P["model"]) if "model" in self.P else None
        # self.sampler = TODO:
        self.interface = self.P["interface"]["kind"](self, **self.P["interface"]) if "interface" in self.P else None
        self._observing = "observe_freq" in self.P and self.P["observe_freq"] > 0
        self._online = "feedback_budget" in self.P and self.P["feedback_budget"] > 0
        if self._online:
            # TODO: More assertions here
            assert self.interface is not None
            assert self._observing
            assert self.P["feedback_freq"] % self.P["observe_freq"] == 0    
            b = self.P["num_episodes_before_freeze"] / self.P["feedback_freq"]
            assert b % 1 == 0
            self._num_batches = int(b)
            self._batch_num = 0
            self._k = 0
            self._n_on_prev_batch = 0
            self._do_save = "save_freq" in self.P and self.P["save_freq"] > 0
            self._do_logs = "log_freq" in self.P and self.P["log_freq"] > 0
            
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
# PREDICTION FUNCTIONS

    def feature_map(self, transitions):
        """
        Map an array of transitions to an array of features.
        """
        if self.features is None: return transitions
        return torch.cat([self.features[f](transitions).reshape(-1,1) for f in self.feature_names], dim=1)

    def reward(self, states, actions=None, next_states=None, transitions=None, features=None, return_params=False):
        """
        Reward function, defined over individual transitions (s,a,s').
        """
        assert self.P["reward_source"] != "extrinsic", "This shouldn't have been called. Unwanted call to pbrl.link(agent)?"
        if "discrete_action_map" in self.P: actions = [self.P["discrete_action_map"][a] for a in actions] 
        transitions = torch.cat([states, actions, next_states], dim=1)
        if self.P["reward_source"] == "oracle": # NOTE: Oracle defined over raw transitions rather than features
            assert not return_params, "Oracle doesn't use normal distribution parameters"
            return torch.tensor(self.interface.oracle(transitions), device=self.device)
        else:
            mu, _, std = self.model(self.feature_map(transitions))        
        if "rune_coef" in self.P: return mu + self.P["rune_coef"] * std
        else: return mu

    def F_ucb_for_pairs(self, trajectories):
        """
        Compute UCB fitness for a sequence of trajectories and sum for all pairs to create a matrix.
        """
        with torch.no_grad(): 
            mu, var = torch.tensor([self.model.fitness(self.feature_map(tr)) for tr in trajectories], device=self.device).T
        F_ucb = mu + self.P["sampler"]["num_std"] * torch.sqrt(var)
        return F_ucb.reshape(-1,1) + F_ucb.reshape(1,-1)

    def Pr_pred(self, trajectory_i, trajectory_j): 
        """
        Predicted probability of trajectory i being preferred to trajectory j.
        """
        raise NotImplementedError()

# ==============================================================================
# FUNCTIONS FOR EXECUTING THE LEARNING PROCESS

    def per_timestep(self, ep, t, state, action, next_state, reward, done, info, extra):     
        """
        Store transition for current timestep.
        """
        if "discrete_action_map" in self.P: action = self.P["discrete_action_map"][action] 
        self._current_ep.append(list(state) + list(action) + list(next_state))
            
    def per_episode(self, ep): 
        """
        NOTE: To ensure video saving, this is completed *after* env.reset() is called for the next episode.
        """     
        self._current_ep = torch.tensor(self._current_ep, device=self.device) # Convert to tensor once appending finished
        logs = {}
        # Log reward sums
        if self.P["reward_source"] == "model": 
            logs["reward_sum_model"] = self.model.fitness(self.feature_map(self._current_ep))[0] 
        if self.interface is not None and self.interface.oracle is not None: 
            logs["reward_sum_oracle"] = sum(self.interface.oracle(self._current_ep))
        # Retain episodes for use in reward inference with a specified frequency
        if self._observing and (ep+1) % self.P["observe_freq"] == 0: 
            self.episodes.append(self._current_ep); n = len(self.episodes)
            Pr_old = self.Pr; self.Pr = torch.full((n, n), float("nan")); self.Pr[:-1,:-1] = Pr_old
        if self._online:
            if (ep+1) % self.P["feedback_freq"] == 0 and (ep+1) <= self.P["num_episodes_before_freeze"]:    
                # Calculate batch size.
                # K = self.P["feedback_budget"] 
                # B = self._num_batches 
                # f = self.P["feedback_freq"] / self.P["observe_freq"] # Number of episodes between feedback batches.
                # c = self.P["scheduling_coef"] 
                # b = self._batch_num # Current batch number.
                # batch_size = int(round((K / B * (1 - c)) + (K * (f * (2*(b+1) - 1) - 1) / (B * (B*f - 1)) * c)))
                assert self.P["scheduling_coef"] == 0
                K = self.P["feedback_budget"] - self._k # Remaining budget
                B = self._num_batches - self._batch_num # Remaining number of batches
                batch_size = int(round(K / B))
                # Gather feedback and update reward function
                self.get_feedback(batch_size=batch_size, ij_min=self._n_on_prev_batch)
                self._batch_num += 1 
                self._n_on_prev_batch = len(self.episodes)
                self.update(history_key=(ep+1))
            logs["feedback_count"] = self._k
            # Periodically log and save out
            if self._do_logs and (ep+1) % self.P["log_freq"] == 0: log(self, history_key=(ep+1))   
            if self._do_save and (ep+1) % self.P["save_freq"] == 0: self.save(history_key=(ep+1))
        self._current_ep = []
        return logs

    def get_feedback(self, ij=None, batch_size=1, ij_min=0): 
        """
        TODO: Make sampler class in similar way to interface class and use __next__ method to loop through.
        """
        if "ucb" in self.P["sampler"]["weight"]: 
            w = self.F_ucb_for_pairs(self.episodes) # Only need to compute once per batch
            if self.P["sampler"]["weight"] == "ucb_r": w = -w # Invert
        elif self.P["sampler"]["weight"] == "uniform": n = len(self.episodes); w = torch.zeros((n, n), device=self.device)
        with self.interface:
            for k in range(batch_size):
                if ij is None:
                    found, i, j, _ = self.select_i_j(w, ij_min=ij_min)
                    if not found: print("=== Fully connected ==="); break
                else: assert batch_size == 1; i, j = ij # Force specified i, j.
                y_ij = self.interface(i, j)
                if y_ij == "esc": print("=== Feedback exited ==="); break
                elif y_ij == "skip": print(f"({i}, {j}) skipped"); continue
                assert 0 <= y_ij <= 1
                self.Pr[i, j] = y_ij
                self.Pr[j, i] = 1 - y_ij
                self._k += 1
                readout = f"{k+1} / {batch_size} ({self._k} / {self.P['feedback_budget']}): P({i} > {j}) = {y_ij}"
                print(readout); self.interface.print("\n"+readout)

    def select_i_j(self, w, ij_min=0):
        """
        Sample a trajectory pair from a weighting matrix subject to constraints.
        """
        if not self.P["sampler"]["constrained"]: raise NotImplementedError()
        n = self.Pr.shape[0]; assert w.shape == (n, n)
        # Enforce non-repeat constraint...
        not_rated = torch.isnan(self.Pr)
        if not_rated.sum() <= n: return False, None, None, None # If have all possible ratings, no more are possible.
        p = w.clone()
        rated = ~not_rated
        p[rated] = float("nan")
        # ...enforce non-identity constraint...
        p.fill_diagonal_(float("nan"))
        # ...enforce connectedness constraint...    
        unconnected = np.argwhere(rated.sum(axis=1) == 0).flatten()
        if len(unconnected) < n: p[unconnected] = float("nan") # (ignore connectedness if first ever rating)
        # ...enforce recency constraint...
        p[:ij_min, :ij_min] = float("nan")
        nans = torch.isnan(p)
        if self.P["sampler"]["probabilistic"]: # NOTE: Approach used in AAMAS paper
            # ...rescale into a probability distribution...
            p -= torch.min(p[~nans]) 
            if torch.nansum(p) == 0: p[~nans] = 1
            p[nans] = 0
            # ...and sample a pair from the distribution
            i, j = np.unravel_index(list(torch.utils.data.WeightedRandomSampler(p.ravel(), num_samples=1))[0], p.shape)
        else: 
            # ...and pick at random from the set of argmax pairs
            argmaxes = np.argwhere(p == torch.max(p[~nans])).T
            i, j = argmaxes[np.random.choice(len(argmaxes))]; i, j = i.item(), j.item()
        # Sense checks
        assert not_rated[i,j]
        if len(unconnected) < n: assert rated[i].sum() > 0 
        assert i >= ij_min or j >= ij_min 
        return True, i, j, p

    def update(self, history_key):
        """
        Update the reward function to reflect the current feedback dataset.
        """
        # Split into training and validation sets
        Pr_train, Pr_val = self.train_val_split()
        # Assemble data structures needed for learning
        A, y, self._connected = self.construct_A_and_y(Pr_train)
        print(f"Connected episodes: {len(self._connected)} / {len(self.episodes)}")
        if len(self._connected) == 0: print("=== None connected ==="); return
        ep_lengths = [len(self.episodes[i]) for i in self._connected]
        # Apply feature mapping to all episodes that are connected to the training set comparison graph
        features = self.feature_map(torch.cat([self.episodes[i] for i in self._connected]))
        # Update the reward function using connected episodes
        self.model.update(history_key, self._connected, ep_lengths, features, A, y)        
        # If applicable, relabel the agent's replay memory using the updated reward function
        self.relabel_memory()  

    def train_val_split(self):
        """
        Split rating matrix into training and validation sets, 
        while keeping comparison graph connected for training set.
        """
        return self.Pr, None

    def construct_A_and_y(self, Pr):
        """
        Construct A and y matrices from a matrix of preference probabilities.
        """
        pairs, y, connected = [], [], set()
        for i, j in np.argwhere(~torch.isnan(Pr)).T: # NOTE: PyTorch v1.10 doesn't have argwhere
            if j < i: pairs.append([i, j]); y.append(Pr[i, j]); connected = connected | {i, j}
        y = torch.tensor(y, device=self.device).float()
        connected = sorted(list(connected))
        A = torch.zeros((len(pairs), len(connected)), device=self.device)
        for l, (i, j) in enumerate(pairs): A[l, [connected.index(i), connected.index(j)]] = torch.tensor([1., -1.])
        return A, y, connected

    def relabel_memory(self): pass  

    def save(self, history_key):
        path = f"run_logs/{self.run_names[-1]}"
        if not os.path.exists(path): os.makedirs(path)
        dump({"params": self.P,
              "Pr": self.Pr,
              "model": self.model
        }, f"{path}/{history_key}.pbrl")


def load(fname):
    """
    Make an instance of PbRLObserver from the information stored by the .save() method.
    NOTE: pbrl.episodes contains the featurised representation of each transition.
    """
    raise NotImplementedError("Needs to be adapted for net models")
    dict = load_jl(fname)
    pbrl = PbrlObserver(dict["params"], features=dict["tree"].space.dim_names[2:])
    pbrl.Pr, pbrl.model = dict["Pr"], dict["model"]
    pbrl.compute_r_and_var()
    A, y, pbrl._connected = pbrl.construct_A_and_y(pbrl.Pr)
    pbrl._ep_fitness_cv = fitness_case_v(A, y, pbrl.P["p_clip"])
    pbrl.episodes = [None for _ in range(pbrl.Pr.shape[0])]
    for i, ep in zip(pbrl._connected, hr.group_along_dim(dict["tree"].space, "ep")):
        assert i == ep[0,0], "pbrl._connected does not match episodes in tree."
        pbrl.episodes[i] = ep[:,2:] # Remove ep and reward columns
    print(f"Loaded {fname}")
    return pbrl
