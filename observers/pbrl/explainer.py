from ...rewards.pbrl.models import maximum_likelihood_fitness, least_squares_fitness

from hyperrectangles.visualise import show_rectangles, show_samples
import os
import numpy as np
from torch import no_grad, cat
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


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

    def show(self): plt.show()

# ==============================================================================
# VISUAL (GLOBAL)

    def plot_loss_correlation(self, history_key, ax=None, c="k"):
        """Correlation between true and proxy loss over pruning sequence."""
        if ax is None: _, ax = plt.subplots()
        ax.set_xlabel("Proxy (variance-based) loss"); ax.set_ylabel("True (preference) loss")
        for key, ls, m, lw in (("grow", "--", None, .5),("prune", "-", ".", 1)):
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
        m_range_split, loss_split, proxy_split = np.array(self.pbrl.model.history[history_key]["grow"]).T
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
        # for i, m in enumerate(mu): plt.scatter(i, m, c="w")
        plt.xticks(range(len(mu)), fontsize=6)

    def plot_alignment(self, history_key, fill_std=False):
        """Scatter plots of oracle vs model returns and rewards."""
        _, axes = plt.subplots(1, 3, figsize=(18, 4))
        _, A, i_list, j_list, y = self.pbrl.graph.make_data_structures()
        connected = set(i_list) | set(j_list)
        raise NotImplementedError("Use epic.epic_with_return")
        rewards, returns = oracle_vs_model_on_graph(self.pbrl.interface.oracle, self.pbrl.model, self.pbrl.graph)
        if len(A):
            # Oracle vs trajectory-level return
            traj_level_returns = maximum_likelihood_fitness(A, y, self.pbrl.model.P["preference_eqn"])[0]
            connected_oracle_returns = [returns[i,0] for i in sorted(connected)]
            axes[0].scatter(connected_oracle_returns, traj_level_returns, s=3, c="k")
            axes[0].set_xlabel("Oracle Return"); axes[0].set_ylabel("Trajectory-level Return")
            # traj_level_returns_old = least_squares_fitness(A, y, 0.1, self.pbrl.model.P["preference_eqn"])[0]
            # axes[3].scatter(connected_oracle_returns, traj_level_returns_old, s=3, c="k")
        # Oracle vs model return
        axes[1].scatter(returns[:,0], returns[:,1], s=3, c=["k" if i in connected else "r" for i in range(len(returns))])
        axes[1].set_xlabel("Oracle Return"); axes[1].set_ylabel("Model Return")
        # Oracle vs model reward
        axes[2].scatter(rewards[:,0], rewards[:,1], s=3, c="k", alpha=0.2)
        axes[2].set_xlabel("Oracle Reward"); axes[2].set_ylabel("Model Reward")
        return True

    def plot_leaf_visitation(self, history_key=None):
        """Heatmap representation of per-episode leaf visitation."""
        plt.figure()
        visits = cat([self.pbrl.model.n(self.pbrl.featuriser(ep["transitions"])).int().unsqueeze(1)
                      for _, ep in self.pbrl.graph.nodes(data=True)], dim=1)
        plt.imshow(visits.T, interpolation="none")

    def plot_preference_graph(self, history_key=None, figsize=(12, 12)):
        raise NotImplementedError("Use self.pbrl.graph.show()")

    def plot_rectangles(self, history_key):
        if self.pbrl.model.tree.root.num_samples == 0: return False
        reward_targets = self.pbrl.model.tree.space.data[:,self.pbrl.model.tree.space.idxify("reward")]
        cmap_lims = (reward_targets.min(), reward_targets.max())
        ax = show_rectangles(self.pbrl.model.tree, [0, 1], attribute=("mean", "reward"), cmap_lims=cmap_lims, maximise=True)
        show_samples(self.pbrl.model.tree.root, [0, 1], colour_dim="reward", ax=ax, cmap_lims=cmap_lims, cbar=False)
        return True

# ==============================================================================
# VISUAL (LOCAL)

    def explain_episode_fitness(self, ep_num):
        from ...rewards.pbrl.models import fitness_case_v
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
        d = node.space.idxify("reward", "ep")
        if ep_wise: d.reverse()
        show_samples(node, vis_dims=[node.split_dim, d[0]], colour_dim=d[1])
        plt.plot([node.split_threshold]*2, node.bb_min[d[0]], c="k")

# ==============================================================================
# TEXTUAL

    def episode_leaf_sequence(self, ep_num):
        """Return the sequence of leaves visited in an episode."""
        return self.pbrl.model.tree.get_leaf_nums(self.pbrl.featuriser(
               self.pbrl.graph.nodes[ep_num]["transitions"]).cpu().numpy())
