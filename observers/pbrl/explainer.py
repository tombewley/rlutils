import os
import numpy as np
from torch import no_grad, cat
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import networkx as nx


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

    def show(self): plt.show()

# ======================================================
# VISUAL (GLOBAL)

    def plot_loss_correlation(self, history_key, ax=None, c="k"):
        """Correlation between true and proxy loss over pruning sequence."""
        if ax is None: _, ax = plt.subplots()
        ax.set_xlabel("Proxy (variance-based) loss"); ax.set_ylabel("True (preference) loss")
        for key, ls, m, lw in (("split", "--", None, .5),("prune", "-", ".", 1)):
            x = self.pbrl.model.history[history_key][key]
            plt.plot([lp for _,_,lp in x], [lt for _,lt,_ in x], c=c, ls=ls, lw=lw, marker=m)
        return ax

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

    def plot_preference_matrix(self, history_key=None):
        """Heatmap representation of preference dataset."""
        plt.figure()
        plt.imshow(self.pbrl.graph.rewind(history_key).preference_matrix,
                   norm=Normalize(0, 1), interpolation="none", cmap="coolwarm_r")

    def plot_sampler_matrix(self, history_key=None):
        """Heatmaps representation of weighting matrix used by sampler."""
        plt.figure()
        for _, _, _, p in self.pbrl.sampler: 
            if p is not None: plt.imshow(p, interpolation="none")
            break

    def plot_fitness_pdfs(self, history_key=None):
        """PDFs of fitness predictions."""
        with no_grad(): mu, var = np.array([self.pbrl.fitness(ep["transitions"]) for _, ep in self.pbrl.graph.nodes(data=True)]).T
        std = np.sqrt(var)
        mn, mx = np.min(mu - 3*std), np.max(mu + 3*std)
        rng = np.arange(mn, mx, (mx-mn)/1000)
        pdfs = np.array([norm.pdf(rng, m, s) for m, s in zip(mu, std)])
        pdfs /= pdfs.max(axis=1).reshape(-1, 1)
        plt.figure()#figsize=(5, 15))
        plt.imshow(pdfs.T, aspect="auto", origin="lower", extent=[-0.5, len(mu)-0.5, mn, mx], interpolation="None")
        for i, m in enumerate(mu): plt.scatter(i, m, c="w")
        plt.xticks(range(len(mu)), fontsize=6)
    
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

    def plot_leaf_visitation(self, history_key=None):
        """Heatmap representation of per-episode leaf visitation."""
        plt.figure()
        visits = cat([self.pbrl.model.n(self.pbrl.featuriser(ep["transitions"])).int().unsqueeze(1)
                      for _,ep in self.pbrl.graph.nodes(data=True)], dim=1)
        plt.imshow(visits.T, interpolation="none")

    def plot_preference_graph(self, history_key=None, figsize=(12, 12)):
        # self.pbrl.graph = nx.DiGraph()
        # n = len(self.episodes)
        # self.pbrl.graph.add_nodes_from(range(n), fitness=np.nan, fitness_cv=np.nan)
        # for i in range(n):
        #     if self.episodes[i] is not None: self.pbrl.graph.nodes[i]["fitness"], _ = self.pbrl.fitness(self.pbrl.episodes[i])
        # for i, f in zip(self._connected, self._ep_fitness_cv):
        #     self.pbrl.graph.nodes[i]["fitness_cv"] = f * len(self.episodes[i])
        # self.pbrl.graph.add_weighted_edges_from([(j, i, self.preferences[i,j]) for i in range(n) for j in range(n) if not np.isnan(self.preferences[i,j])])
        # fitness = list(nx.get_node_attributes(self.pbrl.graph, "fitness").values())
        # fitness_cv = list(nx.get_node_attributes(self.pbrl.graph, "fitness_cv").values())
        # vmin, vmax = min(np.nanmin(fitness), np.nanmin(fitness_cv)), max(np.nanmax(fitness), np.nanmax(fitness_cv))
        g = self.pbrl.graph.rewind(history_key)._graph
        # Plot nodes
        plt.figure(figsize=figsize)
        pos = nx.drawing.nx_agraph.graphviz_layout(g, prog="neato")
        nx.draw_networkx_nodes(g, pos=pos,
            node_size=500,
            # node_color=fitness_cv,
            # cmap="coolwarm_r", vmin=vmin, vmax=vmax
        )
        nx.draw_networkx_nodes(g, pos=pos,
            node_size=250,
            # node_color=fitness,
            # cmap="coolwarm_r", vmin=vmin, vmax=vmax,
            linewidths=1, edgecolors="w"
        )
        nx.draw_networkx_labels(g, pos=pos)
        # Plot edges
        edge_collection = nx.draw_networkx_edges(g, pos=pos,
            node_size=500,
            width=2, arrowsize=20,
            edge_color=list(nx.get_edge_attributes(g, "preference").values()),
            connectionstyle="arc3,rad=0.1",
            edge_cmap=plt.cm.coolwarm_r
        )
        # weights = list(nx.get_edge_attributes(g, "preference").values())
        # for i, e in enumerate(edge_collection): e.set_alpha(weights[i])
        # nx.draw_networkx_edge_labels(g, pos=pos, label_pos=0.4, font_size=6,
        #     edge_labels={(i, j): f"{d['weight']:.2f}" for i, j, d in g.edges(data=True)}
        #     )

# ======================================================
# VISUAL (LOCAL)

    def explain_episode_fitness(self, ep_num):
        from .models import fitness_case_v
        A, y, i_list, j_list, connected = self.pbrl.graph.construct_A_and_y()
        f, d, _ = fitness_case_v(A, y, self.pbrl.model.P["p_clip"])
        other_ep_num, target_fitness = [], []
        for k, (i, j) in enumerate(zip(i_list, j_list)):
            if i == ep_num:
                other_ep_num.append(j)
                target_fitness.append(f[j] + d[k].item())
            elif j == ep_num:
                other_ep_num.append(i)
                target_fitness.append(f[i] - d[k].item())
        plt.scatter(other_ep_num, target_fitness, s=5, c="k")
        plt.plot([min(other_ep_num), max(other_ep_num)], [f[connected.index(ep_num)]]*2)

    def explain_split(self, node, ep_wise=False):
        import hyperrectangles as hr
        d = node.space.idxify("reward", "ep")
        if ep_wise: d.reverse()
        hr.show_samples(node, vis_dims=[node.split_dim, d[0]], colour_dim=d[1])
        plt.plot([node.split_threshold]*2, node.bb_min[d[0]], c="k")

# ======================================================
# TEXTUAL

    def episode_leaf_sequence(self, ep_num):
        """Return the sequence of leaves visited in an episode."""
        return self.pbrl.model.tree.get_leaf_nums(self.pbrl.featuriser(
               self.pbrl.graph.nodes[ep_num]["transitions"]).cpu().numpy())
