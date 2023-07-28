from multiprocessing import reduction
from numpy import array
import torch
torch.set_printoptions(precision=3, edgeitems=10, linewidth=200, sci_mode=False)
from collections import defaultdict
import matplotlib.pyplot as plt
from seaborn import heatmap

from tufte import coloured_2d_plot, stacked_bar_chart
from hyperrectangles.rules import rule
from ..rewards.models import RewardTree as RewardTree1
from reward_trees import RewardTree as RewardTree2


class PetsExplainer:

    dim_order = ("i", "p", "t", "c")
    dim_names = {
        "i" : "Planning iteration",
        "p" : "Particle number",
        "t" : "Planning timestep",
        "c" : "Component"
    }
    P = {}

    def __init__(self, agent):
        self.agent = agent
        self.run_names = []
        self._data = defaultdict(lambda: defaultdict(dict)) # https://stackoverflow.com/a/44992115

    def per_timestep(self, ep, t, state, action, next_state, reward, done, info, extra):
        r = self.run_names[-1]
        try:
            self._data[r][ep][t]
        except:
            pass
        else:
            raise ValueError(f"Already have data for [{r}][{ep}][{t}]")
        self._data[r][ep][t] = {
            "state"        : state,
            "action"       : action,
        }
        if ({"mean", "std", "states", "actions"} - set(extra.keys())) == set():  # i.e. not in random mode
            self._data[r][ep][t] |= {
            "plan_mean"    : extra["mean"],
            "plan_std"     : extra["std"],
            "plan_states"  : extra["states"],
            "plan_actions" : extra["actions"]
        }

        # _, ax = plt.subplots()
        # self.plot_plans(ax=ax, key=(r, ep, t), what="states", dims=(0, 1), iterations=[0, -1], plot_type="line")
        # _, ax = plt.subplots()
        # # self.plot_plans(ax=ax, key=(r, ep, t), what="rewards", plot_type="scatter")
        # self.plot_decomposition(ax=ax, key=(r, ep, t), dims=("t", "c"), iterations=[-1], iterations_f=[0], scale_by_r=True)
        # _, ax = plt.subplots()
        # self.plot_plans(ax=ax, key=(r, ep, t), what="actions", dims=(0, 1), timesteps=[0], plot_type="scatter") #, iterations=[0, -1], s=100)
        # plt.show()

    def per_episode(self, ep):
        if (ep+1) % 500 == 0:
            # self.rollout_review((self.run_names[-1], ep, 0), horizon=50, plot_dims=(16, 24, 25))
            self.rollout_review((self.run_names[-1], ep, 0), horizon=50, plot_dims=(0, 1, 4))
        return {}

    def data(self, *key):
        data = self._data
        for k in key: data = data[k]
        return data

    def feature_names_and_weights(self):
        try:
            try:
                # New RewardTree
                tree = self.agent.model.reward_function.to_hyperrectangles()
                weights = self.agent.model.reward_function.r_mean
            except:
                # Old RewardTree
                tree = self.agent.model.reward_function.forest[0]["tree"]
                weights = self.agent.model.reward_function.forest[0]["r"]
            feature_names = [rule(l) for l in tree.leaves]
        except:
            # Other
            feature_names = self.agent.model.reward_function.feature_names
            weights = self.agent.model.reward_function.weights
        return feature_names, weights

    def plan_features(self, data):
        try:
            raise Exception("Implement featuriser function to unify")
            # New RewardTree
            return torch.cat([f(data["plan_states"][:,:,:-1,:], data["plan_actions"], data["plan_states"][:,:,1:,:]).unsqueeze(-1)
                              for f in self.agent.model.reward_function.features_and_thresholds.keys()], dim=-1)
        except:
            # Old RewardTree
            return self.agent.model.reward_function.featuriser(
                data["plan_states"][:,:,:-1,:], data["plan_actions"], data["plan_states"][:,:,1:,:])

    def plan_rewards(self, data):
        return self.agent.model.reward_function(
            data["plan_states"][:,:,:-1,:], data["plan_actions"], data["plan_states"][:,:,1:,:]).unsqueeze(-1)

    def plan_visits(self, data):
        try:
            # New RewardTree
            # return self.agent.model.reward_function.featuriser(
            return self.agent.model.reward_function.transitions_to_visits(
                data["plan_states"][:,:,:-1,:], data["plan_actions"], data["plan_states"][:,:,1:,:])
        except:
            # Old RewardTree
            tree = self.agent.model.reward_function.forest[0]["tree"]
            features = self.plan_features(data).cpu().numpy()
            return torch.tensor(tree.get_leaf_nums(features, one_hot=True), device=self.agent.device)

    def plot_plans(self, ax, key, what, dims=[0], iterations=None, particles=None, timesteps=None, **kwargs):
        """
        TODO: Allow 3D plotting
        """
        assert what in {"states", "actions", "features", "rewards"}
        assert len(dims) in {1, 2}
        data = self.data(*key)
        data = data["plan_states"]  if what == "states"  else (
               data["plan_actions"] if what == "actions" else (
               self.plan_features(data) if what == "features" else self.plan_rewards(data)))
        data = filter_scale_reduce(data, iterations, particles, timesteps, dims)
        colour_by = torch.arange(data.shape[0]).repeat_interleave(data.shape[1]).cpu().numpy()
        data = torch.flatten(data, end_dim=1).cpu().numpy() # Flatten along particle dimension
        coloured_2d_plot(ax, data=data, colour_by=colour_by, **kwargs,)

    def plot_decomposition(self, ax, dims, key,        iterations=None,   particles=None,   timesteps=None,   components=None,
                           key_f=None, iterations_f=None, particles_f=None, timesteps_f=None, components_f=None,
                           scale_by_r=False, msx_agg=False, transpose=False, plot_type="heatmap", font_size=10, **kwargs):
        """
        
        """
        assert len(dims) in {0, 1, 2}
        reduction_dims = tuple(d for d in self.dim_order if d not in dims)
        feature_names, weights = self.feature_names_and_weights()
        scale = weights if scale_by_r else None

        # ======================================
        # TODO: Pull out

        v_full = self.plan_visits(self.data(*key))
        v = filter_scale_reduce(v_full, iterations, particles, timesteps, components, scale, reduction_dims)
        have_foil = key_f is not None or iterations_f is not None or particles_f is not None or timesteps_f is not None
        if have_foil: # Subtract foil case
            if not(key_f is None or key_f == key): v_full = self.plan_visits(self.data(*key_f))
            v -= filter_scale_reduce(v_full, iterations_f, particles_f, timesteps_f, components_f, scale, reduction_dims)
        squeezed = 0
        for d in self.dim_order:
            if d not in dims:
                v = v.squeeze(self.dim_order.index(d) - squeezed)
                squeezed += 1

        # ======================================

        if msx_agg:
            assert dims == ("t", "c") and scale_by_r and plot_type == "bars"
            diff = v.sum()
            assert diff > 0
            v, ranges, msx_pos, msx_neg = abstract_by_msx(v)
            window_sizes = torch.tensor([rng[1] - rng[0] for rng in ranges])
            rdx_by_t = v.sum(dim=1) * window_sizes
            sign = rdx_by_t.sum().sign()
            msx_pos_by_t, msx_neg_by_t = compute_msx(rdx_by_t)

            r_is_tree = type(self.agent.model.reward_function) in {RewardTree1, RewardTree2}
            print(type(self.agent.model.reward_function))
            def describe(w, c):
                v_unscaled = v[w, c] / scale[c]
                sign = v_unscaled.sign()
                if sign > 0:   change_text = "more likely that" if r_is_tree else "expected increase in"
                elif sign < 0: change_text = "less likely that" if r_is_tree else "expected decrease in"
                else:          raise ValueError
                print(f"      {v_unscaled.abs()} {change_text} {feature_names[c]} (leaf {c+1}, r = {scale[c]})")

            print(f"fact is better by {diff} because")
            for w in msx_pos_by_t:
                # NOTE: A little misleading
                # dca_rule = f"while {rule(tree.find_dca([tree.leaves[c] for c in set(msx_pos[w]) | set(msx_neg[w])]))} "
                dca_rule = ""
                print(f"  {dca_rule}in timesteps {ranges[w][0]} to {ranges[w][1]-1}")
                for c in msx_pos[w]: describe(w, c)
                if len(msx_neg[w]): print("    which outweighs")
                for c in msx_neg[w]: describe(w, c)
            if len(msx_neg_by_t): print("despite")
            for w in msx_neg_by_t:
                print(f"  in timesteps {ranges[w][0]} to {ranges[w][1]-1}")
                for c in msx_pos[w]: describe(w, c)
                if len(msx_neg[w]): print("    which outweighs")
                for c in msx_neg[w]: describe(w, c)

            widths = window_sizes.cpu().numpy()
        else:
            widths = None

        # ======================================

        for _ in range(2 - len(dims)): v = v.unsqueeze(0 if transpose else -1)
        if len(dims) == 2 and self.dim_order.index(dims[0]) > self.dim_order.index(dims[1]):
            v = v.T # Transpose if necessary
        if plot_type == "heatmap": heatmap(v.T, ax=ax, cmap="coolwarm_r", annot=True,
                                           annot_kws={"fontsize": font_size}, cbar=False, center=0, fmt=".2f")
        elif plot_type == "bars":  stacked_bar_chart(ax, v.cpu().numpy(), widths=widths, align="left", **kwargs)
        if len(dims) > 0:
            if len(dims) == 1:
                if not(transpose): ax.set_xlabel(self.dim_names[dims[0]])
                else:
                    ax.set_xticks([])
                    if plot_type == "heatmap": ax.set_ylabel(self.dim_names[dims[0]])
            else:
                ax.set_xlabel(self.dim_names[dims[0]])
                if plot_type == "heatmap": ax.set_ylabel(self.dim_names[dims[1]])
        title = "Decomposed reward" if scale_by_r else "Leaf visitation"
        title += f" (   k = {key}"
        if iterations   is not None: title += f", i = {iterations}"
        if particles    is not None: title += f", p = {particles}"
        if timesteps    is not None: title += f", t = {timesteps}"
        if components   is not None: title += f", c = {components}"
        if have_foil:                title += f"   vs   "
        if key_f        is not None: title += f"k = {key_f}"
        if iterations_f is not None: title += f", i = {iterations_f}"
        if particles_f  is not None: title += f", p = {particles_f}"
        if timesteps_f  is not None: title += f", t = {timesteps_f}"
        if components_f is not None: title += f", c = {components_f}"
        title += "   )"
        ax.set_title(title)

    def rollout_review(self, key, plot_dims, horizon=None, num_rollouts=20):
        """Rollout an episode's action sequence from the initial state and compare true vs simulated state sequences."""
        run_name, ep, t = key
        data = self.data(run_name, ep)
        states = torch.cat([torch.tensor(data[tt]["state"], device=self.agent.device).unsqueeze(0)
                            for tt in range(t, min(len(data), float("inf") if horizon is None else t+horizon))], dim=0)
        actions = torch.cat([torch.tensor(data[tt]["action"], device=self.agent.device).unsqueeze(0)
                             for tt in range(t, min(len(data), float("inf") if horizon is None else t+horizon))], dim=0)
        with torch.no_grad():
            sim_states, _, _ = self.agent.model.rollout(
                states[0].unsqueeze(0), actions=actions.expand(num_rollouts,-1,-1).unsqueeze(2), ensemble_index="ts1_b")
        sim_states = sim_states.squeeze(2)
        plt.figure(); ax = plt.axes(projection="3d")
        for s in sim_states: ax.plot3D(*s[:, plot_dims].T, c="gray", lw=0.5)
        ax.plot3D(*states[:, plot_dims].T, c="k", lw=1)
        ax.scatter3D(*states[:, plot_dims].T, c="k", s=10)
        xlim = ax.get_xlim3d(); ylim = ax.get_ylim3d(); zlim = ax.get_zlim3d()
        ax.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]))
        ax.set_xlabel(f"dim {plot_dims[0]}"); ax.set_ylabel(f"dim {plot_dims[1]}"); ax.set_zlabel(f"dim {plot_dims[2]}")
        plt.show()

def filter_scale_reduce(tn, iterations=None, particles=None, timesteps=None, final_dim=None, scale=None, reduction_dims=tuple()):
    if iterations is not None: tn = tn[iterations]
    if particles is not None:  tn = tn[:,particles]
    if timesteps is not None:  tn = tn[:,:,timesteps]
    if final_dim is not None:  tn = tn[:,:,:,final_dim]
    if scale is not None:      tn = tn.float() * scale[final_dim]
    if "i" in reduction_dims:  tn = tn.float().mean(dim=0, keepdim=True)
    if "p" in reduction_dims:  tn = tn.float().mean(dim=1, keepdim=True)
    if "t" in reduction_dims:  tn = tn.sum(dim=2, keepdim=True) # ***TODO: Discount factor***
    if "c" in reduction_dims:  tn = tn.sum(dim=3, keepdim=True)
    return tn

# def compute_rdx(n_i, n_j, r):
#     """Compute the temporally-decomposed reward difference explanation (RDX)"""
#     n_delta = n_i - n_j
#     nz = n_delta.sum(dim=-2) != 0.
#     return n_delta[...,nz] * r[nz], nz

def compute_msx(rdx):
    """Given an RDX vector, compute a minimal sufficient explanation (MSX+, MSX-)."""
    d = -rdx[rdx < 0.].sum() # Disadvantage d = negated sum of negative elements
    rank = rdx.argsort()
    i = len(rank)
    v = 0. # Just-insufficient advantage v = sum of all but smallest element in MSX+
    for x in reversed(rank):
        if rdx[x] <= 0: break
        i -= 1
        if (v + rdx[x]) > d: break
        v += rdx[x]
    msx_pos = rank[i:].flip([0]) # MSX+ = positive elements required to overcome d
    i = 0
    dd = 0.
    for x in rank:
        if rdx[x] >=0: break
        i += 1
        if (dd - rdx[x]) > v: break
        dd -= rdx[x]
    msx_neg = rank[:i] # MSX- = negative elements required to overcome v and show that all of MSX+ is necessary
    return msx_pos, msx_neg

def abstract_by_msx(rdx_matrix):
    """Given a matrix of RDX vectors, abstract along the first dimension
    so that all members of each window have the same difference sign, MSX+ and MSX-.
    """
    ranges, signs, msx_pos, msx_neg = [], [], [], []
    for t, rdx in enumerate(rdx_matrix):
        sign = rdx.sum().sign()
        if sign >= 0:
            mp, mn = compute_msx(rdx)
        else:
            mp, mn = compute_msx(-rdx)
        if t > 0 and sign == signs[-1] and torch.equal(mp, msx_pos[-1]) and torch.equal(mn, msx_neg[-1]):
            ranges[-1][1] = t + 1
        else:
            ranges.append([t, t + 1])
            signs.append(sign)
            msx_pos.append(mp)
            msx_neg.append(mn)
    return torch.cat([rdx_matrix[rng[0]:rng[1]].mean(dim=0).unsqueeze(0) for rng in ranges], dim=0),\
           ranges,\
           msx_pos,\
           msx_neg

def abstract_by_rank(rdx_matrix):
    """Given a matrix of RDX vectors, abstract along the first dimension
    so that all members of each window have the same component-level difference signs and ranking.
    NOTE: neither strictly looser nor tighter than abstract_by_msx; best choice depends on downstream application.
    """
    ranges, signs, pos, zero, neg = [], [], [], [], []
    for t, rdx in enumerate(rdx_matrix):
        sign = rdx.sum().sign()
        rank = rdx.argsort()
        p, z, n = rank[rdx[rank] > 0], rank[rdx[rank] == 0], rank[rdx[rank] < 0]
        if t > 0 and sign == signs[-1] and torch.equal(p, pos[-1]) and torch.equal(z, zero[-1]) and torch.equal(n, neg[-1]):
            ranges[-1][1] = t + 1
        else:
            ranges.append([t, t + 1])
            signs.append(sign)
            pos.append(p)
            zero.append(z)
            neg.append(n)
    return torch.cat([rdx_matrix[rng[0]:rng[1]].mean(dim=0).unsqueeze(0) for rng in ranges], dim=0),\
           ranges,\
           pos,\
           neg
