import os
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from torch import no_grad


class Explainer:
    def __init__(self, pbrl, P): 
        self.pbrl, self.P = pbrl, P
        self.plot_methods = {name[5:]: method for name, method in Explainer.__dict__.items()
                             if callable(method) and name.startswith("plot_")}
        
    def __call__(self, history_key):
        if self.P["save"]:
            path = f"logs/{self.pbrl.run_names[-1]}"
            if not os.path.exists(path): os.makedirs(path)
        for plot in self.P["plots"]:
            made = self.plot_methods[plot](self, history_key)
            if made and self.P["save"]: plt.savefig(f"{path}/{plot}_{history_key}.png")
        if self.P["show"]: self.show()
        plt.close("all")

        # if history_key in self.pbrl.model.history: # TODO: Hacky
        #     if "tree_diagram" in self.P["plots"]: 
        #         self.pbrl.model.diagram(self.pbrl.model.tree, pred_dims=["reward"], verbose=True, out_name=f"{path}/tree_{history_key}", out_as="png")
        #     if "tree_py" in self.P["plots"]: 
        #         self.pbrl.model.rules(self.pbrl.model.tree, pred_dims="reward", sf=None, out_name=f"{path}/tree_{history_key}.py")
        #     if "tree_rectangles" in self.P["plots"]:
        #         for vis_dims, vis_lims in self.P["plots"]["tree_rectangles"]:
        #             cmap_lims = (self.pbrl.model.r.min().item(), self.pbrl.model.r.max().item())
        #             _, (ax1, ax2) = plt.subplots(1, 2, figsize=(24,8))
        #             self.pbrl.model.rectangles(self.pbrl.model.tree, vis_dims, attribute=("mean", "reward"), vis_lims=vis_lims, cmap_lims=cmap_lims, maximise=True, ax=ax1)
        #             # hr.show_leaf_numbers(self.tree, vis_dims, ax=ax1)
        #             self.pbrl.model.rectangles(self.pbrl.model.tree, vis_dims, attribute=("std", "reward"), vis_lims=vis_lims, maximise=True, ax=ax2)
        #             # hr.show_leaf_numbers(self.tree, vis_dims, ax=ax2)
        #             if False: # Overlay samples
        #                 hr.show_samples(self.tree.root, vis_dims=vis_dims, colour_dim="reward", ax=ax1, cmap_lims=cmap_lims, cbar=False)
        #             plt.savefig(f"{path}/{vis_dims}_{history_key}.png")

    def plot_loss_correlation(self, history_key):
        """Correlation between true and proxy loss."""
        _, ax = plt.subplots()
        ax.set_xlabel("Proxy (variance-based) loss"); ax.set_ylabel("True (preference) loss")
        for history_key in self.pbrl.model.history:
            history_prune = self.pbrl.model.history[history_key]["prune"]
            plt.scatter([lp for _,_,_,_,lp in history_prune], [lt for _,_,_,lt,_ in history_prune], s=3, label=history_key)
        plt.legend()

    def plot_loss_vs_m(self, history_key):
        """Loss as a function of m over splitting/pruning sequence."""
        _, ax1 = plt.subplots()
        ax1.set_xlabel("Number of components (m)"); ax1.set_ylabel("True (preference) loss")
        ax2 = ax1.twinx()
        ax2.set_ylabel("Proxy (variance-based) loss")
        ax2.yaxis.label.set_color("b")
        m_range_split, loss_split, proxy_split = np.array(self.pbrl.model.history[history_key]["split"]).T
        m_range_prune, loss_prune, proxy_prune = np.array(self.pbrl.model.history[history_key]["prune"]).T
        m_final = self.pbrl.model.history[history_key]["m"]
        loss_m_final = loss_prune[np.argwhere(m_range_prune == m_final)[0]]
        ax1.plot(m_range_split, loss_split, c="k", ls="--")
        ax1.plot(m_range_prune, loss_prune, c="k")
        ax1.scatter(m_final, loss_m_final, c="g")
        ax2.plot(m_range_split, proxy_split, c="b", ls="--")
        ax2.plot(m_range_prune, proxy_prune, c="b")
        # Regularisation line
        m_lims = np.array([m_range_prune[0], m_range_prune[-1]])
        ax1.plot(m_lims, loss_m_final - self.pbrl.model.P["alpha"] * (m_lims - m_final), c="g", ls="--", zorder=-1)
        ax1.set_ylim(bottom=0); ax2.set_ylim(bottom=0)

    def plot_preference_matrix(self, history_key):
        """Binary matrix showing which preferences have been obtained."""
        plt.figure()
        plt.imshow(self.pbrl.preference_matrix, norm=Normalize(0, 1), interpolation="none", cmap="coolwarm_r")

    def plot_sampler_matrix(self, history_key):
        """Weighting matrix used by sampler."""
        plt.figure()
        for _, _, _, p in self.pbrl.sampler: 
            if p is not None: plt.imshow(p, interpolation="none")
            break

    def plot_fitness_pdfs(self, history_key):
        """PDFs of fitness predictions."""
        mu, var = np.array([self.pbrl.fitness(ep) for ep in self.pbrl.episodes]).T
        std = np.sqrt(var)
        mn, mx = np.min(mu - 3*std), np.max(mu + 3*std)
        rng = np.arange(mn, mx, (mx-mn)/1000)
        pdfs = np.array([norm.pdf(rng, m, s) for m, s in zip(mu, std)])
        pdfs /= pdfs.max(axis=1).reshape(-1, 1)
        plt.figure(figsize=(5, 15))
        plt.imshow(pdfs, aspect="auto", extent=[mn, mx, len(mu)-0.5, -0.5], interpolation="None")
        for i, m in enumerate(mu): plt.scatter(m, i, c="w")
        plt.yticks(range(len(mu)), fontsize=6)
    
    def plot_alignment(self, history_key, vs="oracle", ax=None, fill_std=False):
        """Decomposed fitness (+/- 1 std) vs a baseline, either:
            - Case V fitness, or
            - Ground truth fitness if an oracle is available    
        """
        if vs == "case_v": 
            raise NotImplementedError()
            baseline, xlabel = self.pbrl.model._ep_fitness_cv, "Case V Fitness"
            ranking = [self.model._connected[i] for i in np.argsort(baseline)]
        elif vs == "oracle":
            assert self.pbrl.interface.oracle is not None
            baseline = np.array([self.pbrl.interface.oracle(ep["transitions"]).sum().item() for _, ep in self.pbrl.graph.nodes(data=True)])
            xlabel = "Oracle Fitness"
            ranking = np.argsort(baseline)
        with no_grad(): mu, var = np.array([self.pbrl.fitness(self.pbrl.graph.nodes[i]["transitions"]) for i in ranking]).T
        if ax is None: _, ax = plt.subplots()
        baseline_sorted = baseline[ranking]
        _, _, _, _, connected = self.pbrl.construct_A_and_y(); connected = set(connected)
        ax.scatter(baseline_sorted, mu, s=3, c=["k" if i in connected else "r" for i in ranking])
        if fill_std:
            std = np.sqrt(var)
            ax.fill_between(baseline_sorted, mu-std, mu+std, color=[.8,.8,.8], zorder=-1, lw=0)
        ax.set_xlabel(xlabel); ax.set_ylabel("Predicted Fitness")
        if False and vs == "oracle":
            baseline_conn, case_v_conn = [], []
            for i in ranking:
                try:
                    c_i = self.pbrl.model._connected.index(i)
                    baseline_conn.append(baseline[i]); case_v_conn.append(self.pbrl.model._ep_fitness_cv[c_i])
                except: continue
            ax2 = ax.twinx()
            ax2.scatter(baseline_conn, case_v_conn, s=3, c="b")
            ax2.set_ylabel("Case V Fitness Fitness")
            ax2.yaxis.label.set_color("b")

    def plot_preference_graph(self, history_key, figsize=(12, 12)):
        # Graph creation.
        self.graph = nx.DiGraph()
        n = len(self.episodes)
        self.graph.add_nodes_from(range(n), fitness=np.nan, fitness_cv=np.nan)
        for i in range(n): 
            if self.episodes[i] is not None: self.graph.nodes[i]["fitness"], _ = self.pbrl.fitness(self.pbrl.episodes[i])
        for i, f in zip(self._connected, self._ep_fitness_cv): 
            self.graph.nodes[i]["fitness_cv"] = f * len(self.episodes[i])
        self.graph.add_weighted_edges_from([(j, i, self.preferences[i,j]) for i in range(n) for j in range(n) if not np.isnan(self.preferences[i,j])])
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

    def show(self): plt.show()
