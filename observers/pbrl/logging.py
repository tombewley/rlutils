import os
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import networkx as nx


def log(self, history_key):
    """Multi-plot generation and saving, to be called periodically after reward function is updated."""
    raise NotImplementedError()
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