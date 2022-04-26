from .featuriser import Featuriser
from ...common.networks import SequentialNetwork
from ...common.utils import reparameterise

import torch
import numpy as np
from scipy.special import xlogy, xlog1py
from scipy.stats import norm as norm_s
from tqdm import tqdm
from gym.spaces.space import Space


norm = torch.distributions.Normal(0, 1)
bce_loss = torch.nn.BCELoss()

# TODO: Split models out into separate files


class RewardModel:
    def __init__(self, P):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.P = P
        self.featuriser = Featuriser(self.P["featuriser"]) if "featuriser" in self.P else None


class RewardNet:
    def __init__(self, P):
        RewardModel.__init__(self, P)
        self.net = SequentialNetwork(device=self.device,
            input_space=[Space((len(self.featuriser.names),))],
            # Mean and log standard deviation
            output_size=2,
            # 3x 256 hidden units used in PEBBLE paper
            code=[(None, 256), "R", (256, 256), "R", (256, 256), "R", (256, None)],
            # 3e-4 used in PEBBLE paper
            lr=3e-4, 
            )
        self.shift, self.scale = 0., 1.

    def __call__(self, transitions=None, features=None, normalise=True):
        if features is None: features = self.featuriser(transitions)
        mu, log_std = reparameterise(self.net(features), clamp=("soft", -2, 2), params=True)
        # NOTE: Scaling up std output helps avoid extreme probabilities
        mu, std = mu.squeeze(1), torch.exp(log_std).squeeze(1) * 100. 
        if normalise:            
            mu, std = (mu - self.shift) / self.scale, std / self.scale  
        var = torch.pow(std, 2.)
        return mu, var, std

    def fitness(self, transitions=None, features=None):
        mu, var, _ = self(transitions, features)
        return mu.sum(), var.sum()

    def update(self, graph, history_key):
        transitions, A, i_list, j_list, y = graph.make_data_structures()
        k_train, n_train = A.shape
        if k_train == 0: print("=== None connected ==="); return {}
        features = [self.featuriser(tr) for tr in transitions] # Featurising up-front may be faster if sampling many batches
        rng = np.random.default_rng()
        losses = []
        for _ in range(self.P["num_batches_per_update"]):
            batch = rng.choice(k_train, size=min(k_train, self.P["batch_size"]), replace=False)
            A_batch, y_batch = A[batch], y[batch]
            abs_A_batch = torch.abs(A_batch)
            in_batch = abs_A_batch.sum(axis=0) > 0
            F_pred, var_pred = torch.zeros(n_train, device=self.device), torch.zeros(n_train, device=self.device)
            for i in range(n_train):
                if in_batch[i]: F_pred[i], var_pred[i] = self.fitness(features=features[i])
            if self.P["preference_eqn"] == "thurstone":
                pair_diff = A_batch @ F_pred
                sigma = torch.sqrt(abs_A_batch @ var_pred)
                y_pred = norm.cdf(pair_diff / sigma)
                loss = bce_loss(y_pred, y_batch) # Binary cross-entropy loss, PEBBLE equation 4
            elif self.P["preference_eqn"] == "bradley-terry":
                # https://github.com/rll-research/BPref/blob/f3ece2ecf04b5d11b276d9bbb19b8004c29429d1/reward_model.py#L142
                F_pairs = torch.vstack([F_pred[[i_list[k], j_list[k]]] for k in batch])
                log_y_pred = torch.nn.functional.log_softmax(F_pairs, dim=1)
                loss = -(torch.column_stack([y_batch, 1-y_batch]) * log_y_pred).sum() / y_batch.shape[0]
            self.net.optimise(loss, retain_graph=False)
            losses.append(loss.item())
        # Normalise rewards to be negative on the training set, with unit standard deviation
        with torch.no_grad(): all_rewards, _, _ = self(features=torch.cat(features), normalise=False)
        self.shift, self.scale = all_rewards.max(), all_rewards.std()
        return {"preference_loss": np.mean(losses)}


class RewardTree(RewardModel):
    def __init__(self, P):
        RewardModel.__init__(self, P)
        # === Lazy import ===
        import hyperrectangles as hr
        self.rules, self.diagram, self.rectangles, self.show_split_quality = \
        hr.rules, hr.diagram, hr.show_rectangles, hr.show_split_quality
        # ===================
        space = hr.Space(dim_names=self.featuriser.names+["ep", "reward"])
        root = hr.node.Node(space, sorted_indices=space.all_sorted_indices)
        self.tree = hr.tree.Tree(
            name="reward_function",
            root=root, 
            split_dims=space.idxify(self.featuriser.names),
            eval_dims=space.idxify(["reward"]),
            )
        self.r = torch.zeros(self.m, device=self.device)
        self.var = torch.zeros(self.m, device=self.device)
        self.history = {} # History of tree modifications

    @property
    def m(self): return len(self.tree.leaves)

    def __call__(self, transitions):
        # NOTE: Awkward torch <-> numpy conversion
        indices = torch.tensor(self.tree.get_leaf_nums(self.featuriser(transitions).cpu().numpy()))
        mu, var = self.r[indices], self.var[indices]
        std = torch.sqrt(var)
        return mu, var, std

    def fitness(self, transitions):
        # https://www.statlect.com/probability-distributions/normal-distribution-linear-combinations
        n = self.n(transitions)
        return n @ self.r, n @ torch.diag(self.var) @ n.T

    def update(self, graph, history_key, reset_tree=True):

        num_repeats = 1
        prune_ratio = 0.5
        post_populate_with_all = False

        for _ in range(num_repeats):
            if reset_tree: self.tree.prune_to(self.tree.root)
            if prune_ratio is not None:
                num_prune = int(round(len(graph.edges) * min(max(0, prune_ratio), 1)))
                num_grow = len(graph.edges) - num_prune
                if num_grow < 1 or num_prune < 1: return {}
                grow_graph, prune_graph = graph.random_connected_subgraph(num_grow)
                self.populate(grow_graph)
                history_grow = self.grow()
                if post_populate_with_all: self.populate(graph)
                history_prune = self.prune(prune_graph)
            else:
                if len(graph.edges) < 1: return {}
                self.populate(graph)
                history_grow, history_prune = self.grow(), self.prune(graph)

            self.history[history_key] = {"grow": history_grow, "prune": history_prune, "m": self.m}
            loss = [l for m,l,_ in self.history[history_key]["prune"] if m == self.m][0]
            print(self.rules(self.tree, pred_dims="reward", sf=5))
            print(loss)

        return {"preference_loss": loss, "num_leaves": self.m}

    def populate(self, graph):
        """
        Populate tree with all episodes in a preference graph. Compute least squares fitness estimates
        for connected episodes, and apply uniform temporal prior to obtain reward targets.
        """
        transitions, A, _, _, y = graph.make_data_structures()
        # NOTE: scaling by episode lengths (making ep fitness correspond to sum not mean) causes weird behaviour
        ep_lengths = [len(tr) for tr in transitions]
        reward_target, _, _ = least_squares_fitness(A, y, self.P["p_clip"], self.P["preference_eqn"]) # * np.mean(ep_lengths) / ep_lengths
        # Populate tree. 
        self.tree.space.data = np.hstack((
            # NOTE: Using index in transition list rather than graph.nodes
            self.featuriser(torch.cat(transitions)).cpu().numpy(),
            np.vstack([[[i, r]] * l for i, (r, l) in enumerate(zip(reward_target, ep_lengths))]),
            ))
        self.tree.populate()
        self.compute_r_and_var()
        return True

    def grow(self):
        """
        Perform best-first splitting until m_max is reached.
        """
        history = []
        def add_to_history(): history.append([self.m, float("nan"), sum(self.tree.gather(("var_sum", "reward"))) / self.tree.root.num_samples])
        add_to_history()
        with tqdm(total=self.P["m_max"], initial=self.m, desc="Splitting") as pbar:
            while self.m < self.P["m_max"] and len(self.tree.split_queue) > 0:
                node = self.tree.split_next_best(self.P["min_samples_leaf"], self.P["num_from_queue"], self.P["store_all_qual"])
                if node is not None: add_to_history(); pbar.update()
        self.compute_r_and_var()
        return history

    def prune(self, graph):
        """
        Recursively prune tree to minimise the (possibly-regularised) preference loss.
        """
        transitions, _, i_list, j_list, y = graph.make_data_structures(unconnected_ok=True)
        y = y.cpu().numpy()
        history = []
        def add_to_history(loss): history.append([self.m, loss, sum(self.tree.gather(("var_sum", "reward"))) / self.tree.root.num_samples])
        tree_before_prune = self.tree.clone()
        prune_indices = []
        mean, var = (np.array(attr) for attr in self.tree.gather(("mean","reward"), ("var","reward")))
        counts = np.vstack([self.n(tr).cpu().numpy() for tr in transitions])
        add_to_history(preference_loss(mean, var, counts, i_list, j_list, y, self.P["preference_eqn"], self.P["loss_func"]))
        with tqdm(total=self.P["m_max"], initial=self.m, desc="Pruning") as pbar:
            while self.m > 1:
                prune_candidates = []
                for i in range(self.m - 1):
                    left, right = self.tree.leaves[i:i+2]
                    if left.parent is right.parent:
                        m = np.delete(mean,   i, axis=0); m[i] = left.parent.mean[-1] # NOTE: Assumes reward is dim -1
                        v = np.delete(var,    i, axis=0); v[i] = left.parent.cov[-1,-1]
                        c = np.delete(counts, i, axis=1); c[:,i] = counts[:,i] + counts[:,i+1]
                        loss = preference_loss(m, v, c, i_list, j_list, y, self.P["preference_eqn"], self.P["loss_func"])
                        prune_candidates.append((i, m, v, c, loss))
                x, mean, var, counts, loss = sorted(prune_candidates, key=lambda cand: cand[4])[0]
                assert self.tree.prune_to(self.tree.leaves[x].parent) == {x, x+1}
                prune_indices.append(x); add_to_history(loss); pbar.update()
        self.tree = tree_before_prune
        # NOTE: Using reversed list to ensure *last* occurrence returned
        optimum = (len(history)-1) - np.argmin([l + (self.P["alpha"] * m) for m,l,_ in reversed(history)])
        for x in prune_indices[:optimum]:
            assert self.tree.prune_to(self.tree.leaves[x].parent) == {x, x+1}
        self.compute_r_and_var()
        return history

    def n(self, transitions):
        assert len(transitions.shape) == 2
        n = torch.zeros(self.m, device=self.device)
        for x in self.tree.get_leaf_nums(self.featuriser(transitions).cpu().numpy()): n[x] += 1
        return n

    def compute_r_and_var(self):
        self.r = torch.tensor(self.tree.gather(("mean","reward")), device=self.device).float()
        self.var = torch.tensor(self.tree.gather(("var","reward")), device=self.device).float()

# ==============================================================================
# UTILITIES

def least_squares_fitness(A, y, p_clip, preference_eqn="thurstone"):
    """
    Construct least fitness estimates under the specified preference equation.
    Uses Morrissey-Gulliksen method for incomplete comparison matrix.
    """
    y = torch.clamp(y, p_clip, 1-p_clip) # Clip to prevent infinite values
    if preference_eqn == "thurstone": d = norm.icdf(y)
    elif preference_eqn == "bradley-terry": raise NotImplementedError()
    f, residuals, _, _ = torch.linalg.lstsq(A.T @ A, A.T @ d, rcond=None) # NOTE: NumPy implementation seems to be more stable
    # f = np.linalg.lstsq((A.T @ A).cpu().numpy(), (A.T @ d).cpu().numpy(), rcond=None)[0]
    return (f - f.max()).cpu().numpy(), d, residuals # NOTE: Shift so that maximum fitness is zero (cost function)

def preference_loss(mean, var, counts, i_list, j_list, y, preference_eqn="thurstone", loss_func="bce"):
    """
    Compute preference loss given vectors of per-component means and variances,
    and a matrix of counts for each episode-component pair.
    """
    assert len(mean.shape) == len(var.shape) == 1 and len(counts.shape) == 2
    assert mean.shape[0] == var.shape[0] == counts.shape[1]
    i_counts, j_counts = counts[i_list], counts[j_list]
    pair_diff = (mean * (i_counts - j_counts)).sum(axis=1)
    pair_var = (var * (i_counts**2 + j_counts**2)).sum(axis=1)
    pair_var[np.logical_and(pair_diff == 0, pair_var == 0)] = 1 # Handle 0/0 case
    if preference_eqn == "thurstone":
        y_pred = norm_s.cdf(pair_diff / np.sqrt(pair_var)) # Div/0 is fine
    elif preference_eqn == "bradley-terry": raise NotImplementedError()
    if loss_func == "bce":
        # Robust binary cross-entropy loss (https://stackoverflow.com/a/50024648)
        loss = (-(xlogy(y, y_pred) + xlog1py(1 - y, -y_pred))).mean()
    elif loss_func == "0-1":
        # Modified 0-1 loss with a central band reserved for "equal" class
        y_shift, y_pred_shift = y - 0.5, y_pred - 0.5
        y_sign =      np.sign(y_shift)      * (np.abs(y_shift) > 0.1)
        y_pred_sign = np.sign(y_pred_shift) * (np.abs(y_pred_shift) > 0.1)
        loss = np.abs(y_sign - y_pred_sign).mean()
    assert not np.isnan(loss)
    return loss
