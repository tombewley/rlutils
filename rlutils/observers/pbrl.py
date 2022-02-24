import hyperrectangles as hr

import os
import numpy as np
np.set_printoptions(precision=3, suppress=True, edgeitems=30, linewidth=100000)   
from scipy.stats import norm
import networkx as nx
from tqdm import tqdm
from joblib import load as load_jl, dump
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


class PbrlObserver:
    def __init__(self, P, features, run_names=None, episodes=None):
        """
        xxx
        """
        self.P = P # Dictionary containing hyperparameters.
        if type(features) == dict: self.feature_names, self.features = list(features.keys()), features
        elif type(features) == list: self.feature_names, self.features = features, None
        self.run_names = run_names if run_names is not None else [] # Order crucial to match with episodes.
        self.load_episodes(episodes if episodes is not None else [])
        self.interface = self.P["interface"]["kind"](self, **self.P["interface"]) if "interface" in self.P else None
        self._do_observe = "observe_freq" in self.P and self.P["observe_freq"] > 0
        self._do_online = "feedback_budget" in self.P and self.P["feedback_budget"] > 0
        if self._do_online:
            # TODO: More assertions here
            assert self.interface is not None
            assert self._do_observe
            assert self.P["feedback_freq"] % self.P["observe_freq"] == 0    
            b = self.P["num_episodes_before_freeze"] / self.P["feedback_freq"]
            assert b % 1 == 0
            self._num_batches = int(b)
            self._batch_num = 0
            self._k = 0
            self._n_on_prev_batch = 0
            self._do_save = "save_freq" in self.P and self.P["save_freq"] > 0
            self._do_logs = "log_freq" in self.P and self.P["log_freq"] > 0
        # Initialise empty tree.
        space = hr.Space(dim_names=["ep", "reward"] + self.feature_names)
        root = hr.node.Node(space, sorted_indices=space.all_sorted_indices) 
        self.tree = hr.tree.Tree(
            name="reward_function", 
            root=root, 
            split_dims=space.idxify(self.feature_names), 
            eval_dims=space.idxify(["reward"])
            )
        # Mean and variance of reward components.  
        self.r, self.var = np.zeros(self.m), np.zeros(self.m) 
        # History of tree modifications.
        self.history = {}

    def link(self, agent):
        """
        NOTE: A little inelegant.
        """
        assert len(agent.memory) == 0, "Agent must be at the start of learning."
        # agent.P["reward"] = self.reward
        agent.memory.__init__(agent.memory.capacity, reward=self.reward, relabel_mode="eager")
        if not agent.memory.lazy_reward: self.relabel_memory = agent.memory.relabel

    @property
    def feedback_count(self): return np.triu(np.invert(np.isnan(self.Pr))).sum()
    @property
    def m(self): return len(self.tree.leaves)

# ==============================================================================
# PREDICTION FUNCTIONS

    def feature_map(self, transitions):
        """
        Map an array of transitions to an array of features.
        """
        if self.features is None: return transitions
        return np.hstack([self.features[f](transitions).reshape(-1,1) for f in self.feature_names])

    def phi(self, transitions):
        """
        Map an array of features to a vector of component indices.
        """
        # if len(transitions.shape) == 1: transitions = transitions.reshape(1,-1) # Handle single.
        return [self.tree.leaves.index(next(iter(self.tree.propagate([None,None]+list(f), mode="max")))) 
                for f in self.feature_map(transitions)]

    def n(self, transitions):
        """
        Map an array of transitions to a vector of component counts.
        """
        n = np.zeros(self.m, dtype=int)
        for x in self.phi(transitions): n[x] += 1
        return n

    def reward(self, states, actions, next_states):
        """
        Reward function, defined over individual transitions. Expects a batch of transitions as three PyTorch tensors.
        """
        assert self.P["reward_source"] != "extrinsic", "This shouldn't have been called. Unwanted call to pbrl.link(agent)?"
        transitions = np.hstack([ # NOTE: PyTorch -> NumPy is quite slow.
            states.cpu().numpy(), 
            [self.P["discrete_action_map"][a] for a in actions] if "discrete_action_map" in self.P else actions.cpu().numpy(), 
            next_states.cpu().numpy()
            ])
        if self.P["reward_source"] == "tree":
            x = self.phi(transitions)
            # TODO: Implement RUNE.
            if "rune_coef" in self.P: return self.r[x] + self.P["rune_coef"] * np.sqrt(self.var[x])
            else: return self.r[x]
        elif self.P["reward_source"] == "oracle":
            return self.interface.oracle(transitions)

    def F(self, trajectory_i, trajectory_j=None):
        """
        Fitness function, defined over trajectories. Returns mean and variance.
        """
        n = (self.n(trajectory_i) if trajectory_j is None else self.n(trajectory_i)-self.n(trajectory_j))
        return [np.matmul(n, self.r), np.matmul(n, np.matmul(np.diag(self.var), n.T))]

    def F_ucb_for_pairs(self, trajectories):
        """
        Compute UCB fitness for a sequence of trajectories and sum for all pairs to create a matrix.
        """
        mu, var = np.array([self.F(tr) for tr in trajectories]).T
        F_ucb = mu + self.P["sampling"]["num_std"] * np.sqrt(var)
        return np.add(F_ucb.reshape(-1,1), F_ucb.reshape(1,-1))

    def Pr_pred(self, trajectory_i, trajectory_j): 
        """
        Predicted probability of trajectory i being preferred to trajectory j.
        """
        F_diff, F_var = self.F(trajectory_i, trajectory_j)
        with np.errstate(divide="ignore"): return norm.cdf(F_diff / np.sqrt(F_var))

# ==============================================================================
# FUNCTIONS FOR EXECUTING THE LEARNING PROCESS

    def load_episodes(self, episodes):
        """
        Load a dataset of episodes and initialise data structures.
        """
        self.episodes = episodes
        self.Pr = np.full((len(episodes), len(episodes)), np.nan)
        self._current_ep = []

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
        # Convert to NumPy now that appending is finished.
        self._current_ep = np.array(self._current_ep) 
        logs = {}
        # Log reward sums.
        if self.P["reward_source"] == "tree": 
            logs["reward_sum_tree"] = self.F(self._current_ep)[0] # NOTE: This overwrites that logged by the environment.
        if self.interface is not None and self.interface.oracle is not None: 
            logs["reward_sum_oracle"] = sum(self.interface.oracle(self._current_ep))
        # Retain episodes for use in reward inference with a specified frequency.
        if self._do_observe and (ep+1) % self.P["observe_freq"] == 0: 
            self.episodes.append(self._current_ep); n = len(self.episodes)
            Pr_old = self.Pr; self.Pr = np.full((n, n), np.nan); self.Pr[:-1,:-1] = Pr_old
        if self._do_online:
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
                self.update(history_key=(ep+1))
                self._batch_num += 1 
                self._n_on_prev_batch = len(self.episodes)
            logs["feedback_count"] = self.feedback_count
            # Periodically save out and plot.
            if self._do_save and (ep+1) % self.P["save_freq"] == 0: self.save(history_key=(ep+1))
            if self._do_logs and (ep+1) % self.P["log_freq"] == 0: self.make_and_save_logs(history_key=(ep+1))   
        self._current_ep = []
        return logs

    def get_feedback(self, ij=None, batch_size=1, ij_min=0): 
        """
        TODO: Make sampler class in similar way to interface class and use __next__ method to loop through.
        """
        if "ucb" in self.P["sampling"]["weight"]: 
            w = self.F_ucb_for_pairs(self.episodes) # Only need to compute once per batch.
            if self.P["sampling"]["weight"] == "ucb_r": w = -w # Invert.
        elif self.P["sampling"]["weight"] == "uniform": n = len(self.episodes); w = np.zeros((n, n))
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
                print(f"{k+1} / {batch_size} ({self._k} / {self.P['feedback_budget']}): P({i} > {j}) = {y_ij}")

    def select_i_j(self, w, ij_min=0):
        """
        Sample a trajectory pair from a weighting matrix subject to constraints.
        """
        if not self.P["sampling"]["constrained"]: raise NotImplementedError()
        n = self.Pr.shape[0]; assert w.shape == (n, n)
        # Enforce non-repeat constraint...
        not_rated = np.isnan(self.Pr)
        if not_rated.sum() <= n: return False, None, None, None # If have all possible ratings, no more are possible.
        p = w.copy()
        rated = np.invert(not_rated)
        p[rated] = np.nan
        # ...enforce non-identity constraint...
        np.fill_diagonal(p, np.nan)
        # ...enforce connectedness constraint...    
        unconnected = np.argwhere(rated.sum(axis=1) == 0).flatten()
        if len(unconnected) < n: p[unconnected] = np.nan # (ignore connectedness if first ever rating)
        # ...enforce recency constraint...
        p[:ij_min, :ij_min] = np.nan
        nans = np.isnan(p)
        if self.P["sampling"]["probabilistic"]: # NOTE: Approach used in AAMAS paper.
            # ...rescale into a probability distribution...
            p -= np.nanmin(p)
            sm = np.nansum(p)
            if sm == 0: p[np.invert(nans)] = 1; p /= np.nansum(p)
            else: p /= sm
            p[nans] = 0
            # ...and sample a pair from the distribution.
            i, j = np.unravel_index(np.random.choice(p.size, p=p.ravel()), p.shape)
        else: 
            # ...and pick at random from the set of argmax pairs.
            argmaxes = np.argwhere(p == np.nanmax(p))
            i, j = argmaxes[np.random.choice(len(argmaxes))]
        # Sense check.
        if len(unconnected) < n: assert np.invert(not_rated[i]).sum() > 0 
        assert i >= ij_min or j >= ij_min 
        return True, i, j, p

    def update(self, history_key, reset_tree=True):
        """
        Update the reward function to reflect the current feedback dataset.
        If reset_tree=True, tree is first pruned back to its root (i.e. start from scratch).
        """
        # Split into training and validation sets.
        Pr_train, Pr_val = train_val_split(self.Pr)
                
        A, d, self._connected = construct_A_and_d(Pr_train, self.P["p_clip"])
        print(f"Connected episodes: {len(self._connected)} / {len(self.episodes)}")
        if len(self._connected) == 0: print("=== None connected ==="); return

        # Compute fitness estimates for episodes that are connected to the training set comparison graph.
        self._ep_fitness_cv = fitness_case_v(A, d)

        # Uniform temporal prior. 
        # NOTE: scaling by episode lengths (making ep fitness correspond to sum not mean) causes weird behaviour.
        ep_length = np.array([len(self.episodes[i]) for i in self._connected])
        reward_target = self._ep_fitness_cv # * ep_length.mean() / ep_length
        
        # Populate tree. 
        self.tree.space.data = np.hstack((
            np.vstack([np.array([[i, r]] * l) for (i, r, l) in zip(self._connected, reward_target, ep_length)]), # Episode number and reward target.
            self.feature_map(np.vstack([self.episodes[i] for i in self._connected]))                             # Feature vector.
            ))
        if reset_tree: self.tree.prune_to(self.tree.root) 
        self.tree.populate()
        num_samples = len(self.tree.space.data)

        # Perform best-first splitting until m_max is reached.
        history_split = []        
        with tqdm(total=self.P["m_max"], initial=self.m, desc="Splitting") as pbar:
            while self.m < self.P["m_max"] and len(self.tree.split_queue) > 0:
                result = self.tree.split_next_best(min_samples_leaf=self.P["min_samples_leaf"]) 
                if result is not None:
                    pbar.update(1)
                    node, dim, threshold = result
                    history_split.append([self.m, node, dim, threshold, None, sum(self.tree.gather(("var_sum", "reward"))) / num_samples])        
        
        # Perform minimal cost complexity pruning until labelling loss is minimised.
        N = np.array([self.n(self.episodes[i]) for i in self._connected])
        tree_before_merge = self.tree.clone()                
        history_merge, parent_num, pruned_nums = [], None, None
        with tqdm(total=self.P["m_max"], initial=self.m, desc="Merging") as pbar:
            while True: 
                # Measure loss.
                r, var, var_sum = self.tree.gather(("mean","reward"),("var","reward"),("var_sum","reward"))
                history_merge.append([self.m, parent_num, pruned_nums,
                    labelling_loss(A, d, N, r, var, self.P["p_clip"]), # True labelling loss.
                    sum(var_sum) / num_samples # Proxy loss: variance.
                    ])
                if self.m <= self.P["m_stop_merge"]: break
                # Perform prune.
                parent_num, pruned_nums = self.tree.prune_mccp()
                pbar.update(-1)
                # Efficiently update N array.
                assert pruned_nums[-1] == pruned_nums[0] + len(pruned_nums)-1
                N[:,pruned_nums[0]] = N[:,pruned_nums].sum(axis=1)
                N = np.delete(N, pruned_nums[1:], axis=1)
                if False: assert (N == np.array([self.n(ep) for ep in self.episodes])).all() # Sense check.     
        
        # Now prune to minimum-loss size.
        # NOTE: Size regularisation applied here; use reversed list to ensure *last* occurrence returned.
        optimum = (len(history_merge)-1) - np.argmin([l + (self.P["alpha"] * m) for m,_,_,l,_ in reversed(history_merge)]) 
        self.tree = tree_before_merge # Reset to pre-merging stage.
        for _, parent_num, pruned_nums_prev, _, _ in history_merge[:optimum+1]: 
            if parent_num is None: continue # First entry of history_merge will have this.
            pruned_nums = self.tree.prune_to(self.tree._get_nodes()[parent_num])
            assert set(pruned_nums_prev) == set(pruned_nums)
        # history_split, history_merge = split_merge_cancel(history_split, history_merge)
        self.history[history_key] = {"split": history_split, "merge": history_merge, "m": self.m}
        print(self.tree.space)
        print(self.tree)
        print(hr.rules(self.tree, pred_dims="reward", sf=5))#, out_name="tree_func"))
                
        # Store updated result.
        self.compute_r_and_var(); self.relabel_memory()  

    def compute_r_and_var(self):
        self.r = np.array(self.tree.gather(("mean","reward"))) 
        self.var = np.array(self.tree.gather(("var","reward")))   

    def relabel_memory(self): pass  

    def save(self, history_key):
        path = f"run_logs/{self.run_names[-1]}"
        if not os.path.exists(path): os.makedirs(path)
        # f = self.features; self.features = None # NOTE: Lambda functions can't be Pickled
        dump({"params": self.P,
              "Pr": self.Pr,
              "tree": self.tree
        }, f"{path}/{history_key}.pbrl")
        # self.features = f

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
            assert self.P["sampling"]["weight"] == "ucb", "Psi matrix only implemented for UCB"
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
        mu, var = np.array([self.F(self.episodes[i]) for i in ranking]).T
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
        mu, var = np.array([self.F(ep) for ep in self.episodes]).T
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
            if self.episodes[i] is not None: self.graph.nodes[i]["fitness"] = self.F(self.episodes[i])[0]
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
    A, d, pbrl._connected = construct_A_and_d(pbrl.Pr, pbrl.P["p_clip"])
    pbrl._ep_fitness_cv = fitness_case_v(A, d)
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
    # pairs = [(i, j) for i, j in np.argwhere(np.triu(np.invert(np.isnan(Pr))))]
    return Pr, None

def construct_A_and_d(Pr, p_clip):
    """
    Construct A and d matrices required by Morrissey-Gulliksen method.
    """
    pairs, y, connected = [], [], set()
    for i, j in np.argwhere(np.logical_not(np.isnan(Pr))):
        if j < i: pairs.append([i, j]); y.append(Pr[i, j]); connected = connected | {i, j}
    connected = sorted(list(connected))
    # Comparison matrix.
    A = np.zeros((len(pairs), len(connected)), dtype=int)
    for l, (i, j) in enumerate(pairs): A[l, [connected.index(i), connected.index(j)]] = [1, -1] 
    # Target vector.
    d = norm.ppf(np.clip(y, p_clip, 1 - p_clip)) 
    return A, d, connected

def fitness_case_v(A, d):
    """
    Construct fitness estimates under Thurstone's Case V model. 
    Uses Morrissey-Gulliksen least squares for incomplete comparison matrix.
    """
    f = np.matmul(np.matmul(np.linalg.pinv(np.matmul(A.T, A)), A.T), d)
    return f - f.max() # NOTE: Shift so that maximum fitness is zero (cost function).

def labelling_loss(A, d, N, r, var, p_clip):
    """
    Loss function l that this algorithm is ultimately trying to minimise.
    """
    N_diff = np.matmul(A, N)
    F_diff = np.matmul(N_diff, r)
    F_std = np.sqrt(np.matmul(N_diff**2, var)) # Faster than actual matrix multiplication N A^T diag(var) A N^T.
    F_std[np.logical_and(F_diff == 0, F_std == 0)] = 1 # Catch 0 / 0 error.
    with np.errstate(divide="ignore"): 
        d_pred = norm.ppf(np.clip(norm.cdf(F_diff / F_std), p_clip, 1-p_clip)) # Clip to prevent infinite values.
        assert not np.isnan(d_pred).any()
    return ((d_pred - d)**2).mean()

def split_merge_cancel(split, merge):
    raise NotImplementedError("Still doesn't work")
    split.reverse()
    for m, (_, siblings, _) in enumerate(merge):
        split_cancel = set()
        subtractions_to_undo = 0
        for s, (_, parent_num, _, _, _) in enumerate(split): 
            if len(siblings) == 1: break
            if parent_num < siblings[-1]:
                siblings = siblings[:-1]
                if parent_num < siblings[0]: 
                    siblings = [siblings[0]-1] + siblings; subtractions_to_undo += 1
                else: 
                    split_cancel.add(s)
                    subtractions_to_undo = 0
                    for ss, (_, later_parent, _, _, _) in enumerate(split[:s]): 
                        if later_parent > siblings[0]: split[ss][1] -= 1 # ; split[ss][0] -= 1
        siblings = [sb+subtractions_to_undo for sb in siblings]
        split = [split[s] for s in range(len(split)) if s not in split_cancel]
        merge[m][1] = siblings
    split.reverse()
    merge = [m for m in merge if len(m[1]) > 1]

    print("====")
    print(split)
    print(merge)

    return split, merge
       
# split = [0,2,5,4,0,3,5,8,10,7]
# merge = [[3,4,5,6,7,8,9,10]]
# [[0, s, None, None, None] for s in split]
# [[None, m, None] for m in merge]

# split_merge_cancel(split, merge)

# ==============================================================================
# INTERFACES

class Interface():
    def __init__(self, pbrl): self.pbrl, self.oracle = pbrl, None
    def __enter__(self): pass
    def __exit__(self, exc_type, exc_value, traceback): pass

class VideoInterface(Interface):
    def __init__(self, pbrl): 
        Interface.__init__(self, pbrl)
        import cv2 # Lazy import
        self.mapping = {81: 1., 83: 0., 32: 0.5, 27: "esc"}

    def __enter__(self):
        self.videos = []
        for rn in self.pbrl.run_names:
            run_videos = sorted([f"video/{rn}/{f}" for f in os.listdir(f"video/{rn}") if ".mp4" in f])
            assert [int(v[-10:-4]) for v in run_videos] == list(range(len(run_videos)))
            self.videos += run_videos
        if len(self.videos) != len(self.pbrl.episodes): 
            assert len(self.videos) == len(self.pbrl.episodes) + 1
            print("Partial video found; ignoring.")                
        cv2.startWindowThread()
        cv2.namedWindow("Trajectory Pairs", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Trajectory Pairs", 1000, 500)

    def __exit__(self, exc_type, exc_value, traceback): 
        cv2.destroyAllWindows()

    def __call__(self, i, j):
        vid_i = cv2.VideoCapture(self.videos[i])
        vid_j = cv2.VideoCapture(self.videos[j])
        while True:
            ret, frame1 = vid_i.read()
            if not ret: vid_i.set(cv2.CAP_PROP_POS_FRAMES, 0); _, frame1 = vid_i.read() # Will get ret = False at the end of the video, so reset.
            ret, frame2 = vid_j.read()
            if not ret: vid_j.set(cv2.CAP_PROP_POS_FRAMES, 0); _, frame2 = vid_j.read()
            if frame1 is None or frame2 is None: raise Exception("Video saving not finished!") 
            cv2.imshow("Trajectory Pairs", np.concatenate((frame1, frame2), axis=1))
            cv2.setWindowProperty("Trajectory Pairs", cv2.WND_PROP_TOPMOST, 1)
            key = cv2.waitKey(10) & 0xFF # https://stackoverflow.com/questions/35372700/whats-0xff-for-in-cv2-waitkey1.                        
            if key in self.mapping: break
        vid_i.release(); vid_j.release()
        return self.mapping[key]

class OracleInterface(Interface):
    """
    Oracle class implementing the five modes of irrationality in the SimTeacher algorithm:

    Lee, K., L. Smith, A. Dragan, and P. Abbeel. 
    "B-Pref: Benchmarking Preference-Based Reinforcement Learning." 
    Neural Information Processing Systems (NeurIPS) (2021).

    (1) "Myopic" recency bias with discount factor gamma
    (2) Query skipping if max(ret_i, ret_j) is below d_skip
        - NOTE: This reduces the effective feedback budget
    (3) Gaussian noise with standard deviation sigma 
        - Analogous to beta in Bradley-Terry model
    (4) Random flipping of P_i with probability epsilon
    (5) Equal preference expression if abs(P_i - 0.5) is below p_equal

    NOTE: Order of implementation here: (1),(2),(3),(4),(5)
    is different to the original paper: (2),(5),(1),(3),(4).

    Additional features:
    (6) Return P_i directly rather than a sample from it - likely to improve performance as gives more information
    TODO:
    (7) Left-right bias *NEED TO RANDOMISE ORDER OF i,j AFTER SAMPLING FOR THIS TO WORK*
    """
    def __init__(self, pbrl, kind=None, oracle=None, gamma=1, sigma=0, d_skip=-np.inf, p_equal=0, epsilon=0, return_P_i=False): 
        Interface.__init__(self, pbrl)
        self.oracle = oracle
        self.gamma, self.sigma, self.d_skip, self.p_equal, self.epsilon, self.return_P_i = gamma, sigma, d_skip, p_equal, epsilon, return_P_i

    def __call__(self, i, j): 
        if type(self.oracle) == list:
            raise NotImplementedError("List-based oracle is deprecated")
            ret_i, ret_j = self.oracle[i], self.oracle[j]
        else:
            ret_i = self.myopic_sum(self.oracle(self.pbrl.episodes[i]))
            ret_j = self.myopic_sum(self.oracle(self.pbrl.episodes[j]))
        if max(ret_i, ret_j) < self.d_skip:  return "skip"
        diff = ret_i - ret_j
        if self.sigma == 0: P_i = 0.5 if diff == 0 else 1. if diff > 0 else 0.
        else:               P_i = norm.cdf(diff / self.sigma)
        if np.random.rand() <= self.epsilon: P_i = 1. - P_i
        if self.return_P_i:                  return P_i
        elif abs(P_i - 0.5) <= self.p_equal: return 0.5
        elif np.random.rand() < P_i:         return 1. 
        else:                                return 0. 

    def myopic_sum(self, rewards):
        if self.gamma == 1: return sum(rewards)
        return sum([r*(self.gamma**t) for t,r in enumerate(reversed(rewards))])
