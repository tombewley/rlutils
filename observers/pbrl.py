import os
import torch
import numpy as np
import networkx as nx
from joblib import load as load_jl, dump
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


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
        # Log reward sums.
        if self.P["reward_source"] == "model": 
            logs["reward_sum_model"] = self.model.fitness(self.feature_map(self._current_ep))[0] 
        if self.interface is not None and self.interface.oracle is not None: 
            logs["reward_sum_oracle"] = sum(self.interface.oracle(self._current_ep))
        # Retain episodes for use in reward inference with a specified frequency.
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
                # Gather feedback and reward function.
                self.get_feedback(batch_size=batch_size, ij_min=self._n_on_prev_batch)
                self._batch_num += 1 
                self._n_on_prev_batch = len(self.episodes)
                self.update(history_key=(ep+1))
            logs["feedback_count"] = self._k
            # Periodically save out and plot.
            if self._do_save and (ep+1) % self.P["save_freq"] == 0: self.save(history_key=(ep+1))
            if self._do_logs and (ep+1) % self.P["log_freq"] == 0: self.make_and_save_logs(history_key=(ep+1))   
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
        Pr_train, Pr_val = train_val_split(self.Pr)
        # Assemble data structures needed for learning
        A, y, self._connected = construct_A_and_y(Pr_train, device=self.device)
        print(f"Connected episodes: {len(self._connected)} / {len(self.episodes)}")
        if len(self._connected) == 0: print("=== None connected ==="); return
        ep_lengths = [len(self.episodes[i]) for i in self._connected]
        # Apply feature mapping to all episodes that are connected to the training set comparison graph
        features = self.feature_map(torch.cat([self.episodes[i] for i in self._connected]))
        # Update the reward function using connected episodes
        self.model.update(history_key, self._connected, ep_lengths, features, A, y)        
        # If applicable, relabel the agent's replay memory using the updated reward function
        self.relabel_memory()  

    def relabel_memory(self): pass  

    def save(self, history_key):
        path = f"run_logs/{self.run_names[-1]}"
        if not os.path.exists(path): os.makedirs(path)
        dump({"params": self.P,
              "Pr": self.Pr,
              "tree": self.tree
        }, f"{path}/{history_key}.pbrl")

# ==============================================================================
# VISUALISATION

    def make_and_save_logs(self, history_key):
        """Multi-plot generation and saving, to be called periodically after reward function is updated."""
        path = f"run_logs/{self.run_names[-1]}"
        if not os.path.exists(path): os.makedirs(path)
        if True: 
            if history_key in self.history: # TODO: Hacky
                self.plot_loss_over_merge(history_key)
                plt.savefig(f"{path}/loss_{history_key}.png")
        if True:
            if history_key in self.history: 
                self.plot_loss_correlation()
                plt.savefig(f"{path}/loss_correlation_{history_key}.png")
        if False: 
            self.plot_comparison_matrix()
            plt.savefig(f"{path}/matrix_{history_key}.png")
        if False: 
            self.plot_alignment()
            plt.savefig(f"{path}/alignment_{history_key}.png")
        if False: 
            self.plot_fitness_pdfs()
            plt.savefig(f"{path}/pdfs_{history_key}.png")
        if False:
            if history_key in self.history: 
                for vis_dims, vis_lims in [([2, 3], None)]:
                    self.plot_rectangles(vis_dims, vis_lims)
                    plt.savefig(f"{path}/{vis_dims}_{history_key}.png")
        if False: # Psi_matrix
            assert self.P["sampler"]["weight"] == "ucb", "Psi matrix only implemented for UCB"
            _, _, _, p = self.select_i_j(self.F_ucb_for_pairs(self.episodes), ij_min=self._n_on_prev_batch)
            plt.figure()
            plt.imshow(p, interpolation="none")
            plt.savefig(f"{path}/psi_matrix_{history_key}.png")
        if True: # Tree as diagram
            if history_key in self.history: 
                hr.diagram(self.tree, pred_dims=["reward"], verbose=True, out_name=f"{path}/tree_{history_key}", out_as="png")
        if True: # Tree as Python function
            if history_key in self.history: 
                hr.rules(self.tree, pred_dims="reward", sf=None, out_name=f"{path}/tree_{history_key}.py")
        
        plt.close("all")

    def plot_loss_over_merge(self, history_key):
        """Loss as a function of m over merging sequence."""
        history_merge, m = self.history[history_key]["merge"], self.history[history_key]["m"]
        m_range = [mm for mm,_,_,_,_ in history_merge]
        loss_m = history_merge[m_range.index(m)][3]
        _, ax1 = plt.subplots()
        ax1.set_xlabel("Number of components (m)"); ax1.set_ylabel("True (labelling) loss")
        ax1.plot(m_range, [l for _,_,_,l,_ in history_merge], c="k") 
        ax1.scatter(m, loss_m, c="g") 
        # Regularisation line
        m_lims = np.array([m_range[0], m_range[-1]])
        ax1.plot(m_lims, loss_m - self.P["alpha"] * (m_lims - m), c="g", ls="--", zorder=-1) 
        ax1.set_ylim(bottom=0)
        ax2 = ax1.twinx()
        ax2.set_ylabel("Proxy (variance-based) loss")
        ax2.yaxis.label.set_color("b")
        ax2.plot(m_range, [l for _,_,_,_,l in history_merge], c="b") 
        ax2.set_ylim(bottom=0)

    def plot_loss_correlation(self):
        """Correlation between true and proxy loss."""
        _, ax = plt.subplots()
        ax.set_xlabel("Proxy (variance-based) loss"); ax.set_ylabel("True (labelling) loss")
        for history_key in self.history:
            history_merge = self.history[history_key]["merge"]
            plt.scatter([lp for _,_,_,_,lp in history_merge], [lt for _,_,_,lt,_ in history_merge], s=3, label=history_key)
        plt.legend()

    def plot_comparison_matrix(self):
        """Binary matrix showing which comparisons have been made."""
        plt.figure()
        plt.imshow(np.invert(np.isnan(self.Pr)), norm=Normalize(0, 1), interpolation="none")

    def plot_alignment(self, vs="ground_truth", ax=None):
        """Decomposed fitness (+/- 1 std) vs a baseline, either:
            - Case V fitness, or
            - Ground truth fitness if an oracle is available    
        """
        print("NOTE: plot_alignment() is expensive!")
        if vs == "case_v": 
            baseline, xlabel = self._ep_fitness_cv, "Case V Fitness"
            ranking = [self._connected[i] for i in np.argsort(baseline)]
        elif vs == "ground_truth":
            assert self.interface.oracle is not None
            if type(self.interface.oracle) == list: baseline = self.interface.oracle
            else: baseline = [sum(self.interface.oracle(ep)) for ep in self.episodes]
            xlabel = "Oracle Fitness"
            ranking = np.argsort(baseline)
        mu, var = np.array([self.model.fitness(self.feature_map(self.episodes[i])) for i in ranking]).T
        std = np.sqrt(var)
        if ax is None: _, ax = plt.subplots()
        baseline_sorted = sorted(baseline)
        connected_set = set(self._connected)
        ax.scatter(baseline_sorted, mu, s=3, c=["k" if i in connected_set else "r" for i in ranking])
        ax.fill_between(baseline_sorted, mu-std, mu+std, color=[.8,.8,.8], zorder=-1, lw=0)
        ax.set_xlabel(xlabel); ax.set_ylabel("Predicted Fitness")
        if False and vs == "ground_truth":
            baseline_conn, case_v_conn = [], []
            for i in ranking:
                try:
                    c_i = self._connected.index(i)
                    baseline_conn.append(baseline[i]); case_v_conn.append(self._ep_fitness_cv[c_i])
                except: continue
            ax2 = ax.twinx()
            ax2.scatter(baseline_conn, case_v_conn, s=3, c="b")
            ax2.set_ylabel("Case V Fitness Fitness")
            ax2.yaxis.label.set_color("b")

    def plot_fitness_pdfs(self):
        """PDFs of fitness predictions."""
        mu, var = np.array([self.model.fitness(self.feature_map(ep)) for ep in self.episodes]).T
        mn, mx = np.min(mu - 3*var**.5), np.max(mu + 3*var**.5)
        rng = np.arange(mn, mx, (mx-mn)/1000)
        P = np.array([norm.pdf(rng, m, v**.5) for m, v in zip(mu, var)])
        P /= P.max(axis=1).reshape(-1, 1)
        plt.figure(figsize=(5, 15))
        # for p in P: plt.plot(rng, p)
        plt.imshow(P, 
        aspect="auto", extent=[mn, mx, len(self.episodes)-0.5, -0.5], interpolation="None")
        plt.yticks(range(len(self.episodes)), fontsize=6)

    def plot_rectangles(self, vis_dims, vis_lims):
        """Projected hyperrectangles showing component means and standard deviations."""
        cmap_lims = (self.r.min(), self.r.max())
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(24,8))
        hr.show_rectangles(self.tree, vis_dims, attribute=("mean", "reward"), vis_lims=vis_lims, cmap_lims=cmap_lims, maximise=True, ax=ax1)
        hr.show_leaf_numbers(self.tree, vis_dims, ax=ax1)
        hr.show_rectangles(self.tree, vis_dims, attribute=("std", "reward"), vis_lims=vis_lims, maximise=True, ax=ax2)
        hr.show_leaf_numbers(self.tree, vis_dims, ax=ax2)
        if True: # Overlay samples.
            hr.show_samples(self.tree.root, vis_dims=vis_dims, colour_dim="reward", ax=ax1, cmap_lims=cmap_lims, cbar=False)
        return ax1, ax2

    def plot_comparison_graph(self, figsize=(12, 12)):
        # Graph creation.
        self.graph = nx.DiGraph()
        n = len(self.episodes)
        self.graph.add_nodes_from(range(n), fitness=np.nan, fitness_cv=np.nan)
        for i in range(n): 
            if self.episodes[i] is not None: self.graph.nodes[i]["fitness"] = self.model.fitness(self.feature_map(self.episodes[i]))[0]
        for i, f in zip(self._connected, self._ep_fitness_cv): 
            self.graph.nodes[i]["fitness_cv"] = f * len(self.episodes[i])
        self.graph.add_weighted_edges_from([(j, i, self.Pr[i,j]) for i in range(n) for j in range(n) if not np.isnan(self.Pr[i,j])])
        # Graph plotting.
        plt.figure(figsize=figsize)
        fitness = list(nx.get_node_attributes(self.graph, "fitness").values())
        fitness_cv = list(nx.get_node_attributes(self.graph, "fitness_cv").values())
        vmin, vmax = min(np.nanmin(fitness), np.nanmin(fitness_cv)), max(np.nanmax(fitness), np.nanmax(fitness_cv))
        pos = nx.drawing.nx_agraph.graphviz_layout(self.graph, prog="neato")
        nx.draw_networkx_nodes(self.graph, pos=pos, 
            node_size=500,
            node_color=fitness_cv,
            cmap="coolwarm_r", vmin=vmin, vmax=vmax
        )
        nx.draw_networkx_nodes(self.graph, pos=pos, 
            node_size=250,
            node_color=fitness,
            cmap="coolwarm_r", vmin=vmin, vmax=vmax,
            linewidths=1, edgecolors="w"
        )
        edge_collection = nx.draw_networkx_edges(self.graph, pos=pos, node_size=500, connectionstyle="arc3,rad=0.1")
        weights = list(nx.get_edge_attributes(self.graph, "weight").values())
        for i, e in enumerate(edge_collection): e.set_alpha(weights[i])
        nx.draw_networkx_labels(self.graph, pos=pos)
        # nx.draw_networkx_edge_labels(self.graph, pos=pos, label_pos=0.4, font_size=6,
        #     edge_labels={(i, j): f"{d['weight']:.2f}" for i, j, d in self.graph.edges(data=True)}
        #     )

# ==============================================================================
# UTILITIES

def load(fname):
    """
    Make an instance of PbRLObserver from the information stored by the .save() method.
    NOTE: pbrl.episodes contains the featurised representation of each transition.
    """
    dict = load_jl(fname)
    pbrl = PbrlObserver(dict["params"], features=dict["tree"].space.dim_names[2:])
    pbrl.Pr, pbrl.tree = dict["Pr"], dict["tree"]
    pbrl.compute_r_and_var()
    A, y, pbrl._connected = construct_A_and_y(pbrl.Pr, device=pbrl.device)
    pbrl._ep_fitness_cv = fitness_case_v(A, y, pbrl.P["p_clip"])
    pbrl.episodes = [None for _ in range(pbrl.Pr.shape[0])]
    for i, ep in zip(pbrl._connected, hr.group_along_dim(dict["tree"].space, "ep")):
        assert i == ep[0,0], "pbrl._connected does not match episodes in tree."
        pbrl.episodes[i] = ep[:,2:] # Remove ep and reward columns
    print(f"Loaded {fname}")
    return pbrl

def train_val_split(Pr):
    """
    Split rating matrix into training and validation sets, 
    while keeping comparison graph connected for training set.
    """
    return Pr, None

def construct_A_and_y(Pr, device):
    """
    Construct A and y matrices from a matrix of preference probabilities.
    """
    pairs, y, connected = [], [], set()
    for i, j in np.argwhere(~torch.isnan(Pr)).T: # NOTE: PyTorch v1.10 doesn't have argwhere
        if j < i: pairs.append([i, j]); y.append(Pr[i, j]); connected = connected | {i, j}
    y = torch.tensor(y, device=device).float()
    connected = sorted(list(connected))
    A = torch.zeros((len(pairs), len(connected)), device=device)
    for l, (i, j) in enumerate(pairs): A[l, [connected.index(i), connected.index(j)]] = torch.tensor([1., -1.])
    return A, y, connected
