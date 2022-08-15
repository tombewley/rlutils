from ..common.featuriser import Featuriser
from ..common.networks import SequentialNetwork
from ..common.utils import reparameterise
from .evaluate import bt_loss_inner

import torch
import numpy as np
from scipy.stats import norm as norm_s
from tqdm import tqdm
from gym.spaces.space import Space
import hyperrectangles as hr


norm = torch.distributions.Normal(0, 1)
mse_loss = torch.nn.MSELoss()
bce_loss = torch.nn.BCELoss()

# TODO: Split models out into separate files


class RewardModel:
    def __init__(self, P):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.P = P
        self.featuriser = Featuriser(self.P["featuriser"])

    def __call__(self, states, actions, next_states):
        mu, _, std = self._call_inner(self.featuriser(states, actions, next_states))
        if "rune_coef" in self.P: return mu + self.P["rune_coef"] * std
        else: return mu


class RewardNet(RewardModel):
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

    def _call_inner(self, features, normalise=True):
        mu, log_std = reparameterise(self.net(features), clamp=("soft", -2, 2), params=True)
        # NOTE: Scaling up std output helps avoid extreme probabilities
        mu, std = mu.squeeze(-1), torch.exp(log_std).squeeze(-1) * 100.
        if normalise: mu, std = (mu - self.shift) / self.scale, std / self.scale
        return mu, torch.pow(std, 2.), std

    def fitness(self, states=None, actions=None, next_states=None, features=None):
        if features is None: features = self.featuriser(states, actions, next_states)
        mu, var, _ = self._call_inner(features)
        return mu.sum(), var.sum()

    def update(self, graph, mode, history_key=None):
        if mode == "reward":
            states, actions, next_states = graph.states, graph.actions, graph.next_states
            rewards = torch.cat([ep["oracle_rewards"] for _,ep in graph.nodes(data=True)])
            k_train = len(rewards)
        elif mode == "return":
            states, actions, next_states = graph.states, graph.actions, graph.next_states
            returns = torch.tensor([ep["oracle_return"] for _,ep in graph.nodes(data=True)], device=self.device)
            k_train = len(returns)
        elif mode == "preference":
            states, actions, next_states, A, i_list, j_list, y = graph.preference_data_structures()
            k_train, n_train = A.shape
        if k_train == 0: print("=== No data for model update ==="); return {}
        # Featurising up-front may be faster if sampling many batches
        features = [self.featuriser(s, a, ns) for s, a, ns in zip(states, actions, next_states)]
        features_cat = torch.cat(features)
        rng = np.random.default_rng()
        for _ in range(self.P["num_batches_per_update"]):
            batch = rng.choice(k_train, size=min(k_train, self.P["batch_size"]), replace=False)
            if mode == "reward":
                loss = mse_loss(self.net(features_cat[batch])[:,0], rewards[batch])
            elif mode == "return":
                loss = mse_loss(torch.stack([self.net(features[i])[:,0].sum() for i in batch]), returns[batch])
            elif mode == "preference":
                assert len(features) == n_train, "Not using subgraph s,a,ns?"
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
        # Normalise rewards to be negative on the training set, with unit standard deviation
        with torch.no_grad(): all_rewards, _, _ = self._call_inner(features_cat, normalise=False)
        self.shift, self.scale = all_rewards.max(), all_rewards.std()
        return {}


class RewardTree(RewardModel):
    def __init__(self, P):
        RewardModel.__init__(self, P)
        self.tree = self.make_tree()
        self.history = {} # History of tree modifications
        # Bring in some convenient visualisation methods
        self.rules, self.diagram, self.rectangles, self.show_split_quality = \
        hr.rules, hr.diagram, hr.show_rectangles, hr.show_split_quality

    def _call_inner(self, features):
        # NOTE: Awkward torch <-> numpy conversion
        indices = torch.tensor(self.tree.get_leaf_nums(features.cpu().numpy()), device=self.device)
        mu, var = self.r[indices], self.var[indices]
        std = torch.sqrt(var)
        return mu, var, std

    @property
    def tree(self): return self._tree
    @tree.setter
    def tree(self, tree):
        self._tree = tree
        self.r = torch.tensor(tree.gather(("mean","reward")), device=self.device).float()
        self.var = torch.tensor(tree.gather(("var","reward")), device=self.device).float()
        self.r[torch.isnan(self.r)] = 0. # NOTE: Reward defaults to 0. in the absence of data.
        assert not(torch.isnan(self.r).any() or torch.isnan(self.var).any())

    def make_tree(self, seed_func=None, name="reward_function"):
        space = hr.Space(dim_names=self.featuriser.names+["ep", "reward"])
        split_dims = space.idxify(self.featuriser.names)
        r_d = space.idxify("reward")
        if seed_func is not None:
            tree = space.tree_from_func(name=name, func=seed_func)
            tree.split_dims, tree.eval_dims = split_dims, [r_d]
            # NOTE: Manually setting per-leaf reward predictions. Use with care!
            for leaf in tree.leaves: leaf.mean[r_d] = leaf.meta["return"]
            return tree
        else: return hr.tree.Tree(name=name, root=hr.node.Node(space),
                                  split_dims=split_dims, eval_dims=[r_d])

    def fitness(self, states, actions, next_states):
        # https://www.statlect.com/probability-distributions/normal-distribution-linear-combinations
        n = self.n(self.tree, states, actions, next_states)
        return n @ self.r, n @ torch.diag(self.var) @ n.T

    def update(self, graph, mode, history_key, reset_tree=True):
        assert reset_tree; trees = [self.make_tree() for _ in range(self.P["trees_per_update"])]
        histories = [{} for _ in range(self.P["trees_per_update"])]
        # Set factor for scaling rewards when computing preference loss
        self._mean_ep_length = np.mean(graph.ep_lengths)
        for tree, history in zip(trees, histories):
            if self.P["prune_ratio"] is not None:
                raise NotImplementedError("Not implemented for mode in {reward, return}; implement ensembling using another class")

                if self.P["nodewise_partition"]:
                    num_prune = int(round(len(graph) * min(max(0, self.P["prune_ratio"]), 1)))
                    num_grow = len(graph) - num_prune
                    if num_grow < 1 or num_prune < 1: return {}
                    grow_graph, prune_graph, eval_graph = graph.random_nodewise_connected_subgraph(num_grow, partitioned=True)
                    assert not (set(grow_graph.nodes) & set(prune_graph.nodes))
                    assert set(grow_graph.edges) | set(prune_graph.edges) | set(eval_graph.edges) == set(graph.edges)
                    if len(grow_graph.edges) < 1 or len(prune_graph.edges) < 1: return {}
                else:
                    num_prune = int(round(len(graph.edges) * min(max(0, self.P["prune_ratio"]), 1)))
                    num_grow = len(graph.edges) - num_prune
                    if num_grow < 1 or num_prune < 1: return {}
                    grow_graph, prune_graph = graph.random_connected_subgraph(num_grow)
                    eval_graph = prune_graph # NOTE: eval_graph is the same as the prune graph
                    assert set(grow_graph.edges) | set(prune_graph.edges) == set(graph.edges)
                assert not (set(grow_graph.edges) & set(prune_graph.edges))

                print(f"Grow graph: {len(grow_graph.edges)} preferences between {len(grow_graph)} episodes")
                print(f"Prune graph: {len(prune_graph.edges)} preferences between {len(prune_graph)} episodes")
                print(f"Eval graph: {len(eval_graph.edges)} preferences between {len(eval_graph)} episodes")

                # Grow using the grow_graph
                self.populate(tree, grow_graph)
                history["grow"] = self.grow(tree)
                if self.P["post_populate_with_all"]: self.populate(tree, graph)
                # Prune using the prune_graph
                history["prune"] = self.prune(tree, prune_graph)#, eval_graph)
                # Evaluate using the eval_graph
                history["loss"] = self.preference_loss(*self.make_loss_data_structures(tree, eval_graph))
            else:
                assert self.P["trees_per_update"] == 1
                if not self.populate(tree, graph, mode=mode):
                    print("=== No data for model update ==="); return {}
                history["grow"] = self.grow(tree)
                history["prune"], history["loss"] = self.prune(tree, graph) if mode == "preference" else self.prune_mccp(tree)

        i, loss = min([(i, h["loss"]) for i, h in enumerate(histories)], key=lambda x:x[1])
        self.tree = trees[i] # NOTE: This triggers tree.setter
        self.history[history_key] = histories[i]

        print([(_i, h["loss"]) for _i, h in enumerate(histories)])
        print(i)
        print(self.rules(self.tree, pred_dims="reward", sf=5, dims_as_indices=False))

        self._mean_ep_length = None
        return {"num_leaves": len(self.tree)}

    def populate(self, tree, graph, mode):
        """
        Populate tree using a preference graph, computing reward targets differently depending on mode.
        """
        # ===========================
        # COMMON WITH NET, TODO: Put into graph.make_data_structures?
        if mode == "reward":
            states, actions, next_states = graph.states, graph.actions, graph.next_states
            rewards = torch.cat([ep["oracle_rewards"] for _,ep in graph.nodes(data=True)])
            k_train = len(rewards)
        elif mode == "return":
            states, actions, next_states = graph.states, graph.actions, graph.next_states
            returns = [ep["oracle_return"] for _,ep in graph.nodes(data=True)]
            k_train = len(returns)
        elif mode == "preference":
            states, actions, next_states, A, _, _, y = graph.preference_data_structures()
            k_train  = len(A)
        if k_train == 0: return False
        features = [self.featuriser(s, a, ns) for s, a, ns in zip(states, actions, next_states)]
        features_cat = torch.cat(features)
        # ==========================

        ep_lengths = [len(s) for s in states] # NOTE: Unconnected episodes have been removed
        ep_nums = np.vstack([[i] * l for i, l in enumerate(ep_lengths)])
        if mode == "reward": rewards = rewards.cpu().numpy()
        else:
            if mode == "preference":
                print("Computing maximum likelihood returns...")
                returns, loss = maximum_likelihood_fitness(A, y, self.P["preference_eqn"])
                returns *= self._mean_ep_length
                print(f"Done (loss = {loss})")
            # NOTE: scaling by episode lengths (making ep fitness correspond to sum not mean) causes weird behaviour
            rewards = np.vstack([[g / l] * l for g, l in zip(returns, ep_lengths)])
        # Populate space, then the tree itself
        tree.space.data = np.hstack((features_cat.cpu().numpy(), ep_nums.reshape(-1,1), rewards.reshape(-1,1)))
        tree.populate()
        return True

    def grow(self, tree):
        """
        Given a populated tree, perform best-first variance-based splitting until m_max is reached.
        """
        history = []
        with tqdm(total=self.P["m_max"], initial=len(tree), desc="Splitting") as pbar:
            while len(tree) < self.P["m_max"] and len(tree.split_queue) > 0:
                node = tree.split_next_best(self.P["min_samples_leaf"], self.P["num_from_queue"], self.P["store_all_qual"])
                if node is not None:
                    history.append({
                        "split_node":      tree.leaves.index(node.left),
                        "split_dim":       node.split_dim,
                        "split_threshold": node.split_threshold
                    })
                    pbar.update()
        return history

    def prune(self, tree, graph, eval_graph=None):
        """
        Recursively prune tree to minimise the (possibly-regularised) preference loss on the given graph.
        Optionally use a second eval_graph to determine the stopping condition.
        """
        mean, var, counts, i_list, j_list, y = self.make_loss_data_structures(tree, graph)
        if eval_graph is None:
            losses = [self.preference_loss(mean, var, counts, i_list, j_list, y).item() + (self.P["alpha"] * len(tree))]
        else:
            _, _, eval_counts, eval_i_list, eval_j_list, eval_y = self.make_loss_data_structures(tree, eval_graph)
            losses = [self.preference_loss(mean, var, eval_counts, eval_i_list, eval_j_list, eval_y).item() + (self.P["alpha"] * len(tree))]
        subtree = tree.clone()
        r_d = tree.space.idxify("reward")
        history = []
        with tqdm(total=len(tree), initial=len(tree), desc="Pruning") as pbar:
            while len(subtree) > 1:
                prune_candidates = []
                for x, _ in subtree.siblings:
                    parent = tree.leaves[x].parent
                    m = np.delete(mean,   x, axis=0); m[x] = parent.mean[r_d]
                    v = np.delete(var,    x, axis=0); v[x] = parent.cov[r_d,r_d]
                    c = np.delete(counts, x, axis=1); c[:,x] = counts[:,x] + counts[:,x+1]
                    loss = self.preference_loss(m, v, c, i_list, j_list, y)
                    prune_candidates.append((x, m, v, c, loss))
                x, mean, var, counts, loss = sorted(prune_candidates, key=lambda cand: cand[4])[0]
                assert subtree.prune_to(subtree.leaves[x].parent) == {x, x+1}
                if eval_graph is not None:
                    eval_counts[:,x] += eval_counts[:,x+1]; eval_counts = np.delete(eval_counts, x+1, axis=1)
                    loss = self.preference_loss(mean, var, eval_counts, eval_i_list, eval_j_list, eval_y)
                history.append({"prune_nodes": {x, x+1}})
                losses.append(loss.item() + (self.P["alpha"] * len(subtree)))
                pbar.update(-1)
        # NOTE: Using reversed list to ensure *last* occurrence returned
        optimum = (len(losses)-1) - np.argmin(list(reversed(losses)))
        for h in history[:optimum]:
            x = min(h["prune_nodes"])
            assert tree.prune_to(tree.leaves[x].parent) == {x, x+1}
        assert len(tree) == (len(losses) - optimum)
        return history, losses[optimum]
    
    def prune_mccp(self, tree):
        raise NotImplementedError

    def make_loss_data_structures(self, tree, graph):
        states, actions, next_states, _, i_list, j_list, y = graph.preference_data_structures(unconnected_ok=True)
        mean, var = (np.array(attr) for attr in tree.gather(("mean","reward"), ("var","reward")))
        counts = np.vstack([self.n(tree, s, a, ns).cpu().numpy() for s, a, ns in zip(states, actions, next_states)])
        return mean, var, counts, i_list, j_list, y

    def preference_loss(self, mean, var, counts, i_list, j_list, y):
        """
        Compute preference loss given vectors of per-component means and variances,
        and a matrix of counts for each episode-component pair.
        """
        assert len(mean.shape) == len(var.shape) == 1 and len(counts.shape) == 2
        assert mean.shape[0] == var.shape[0] == counts.shape[1]
        i_counts, j_counts = counts[i_list], counts[j_list]
        pair_diff = (mean * (i_counts - j_counts)).sum(axis=1)
        if self.P["preference_eqn"] == "thurstone":
            raise NotImplementedError("Apply scale factor")
            pair_var = (var * (i_counts**2 + j_counts**2)).sum(axis=1)
            pair_var[np.logical_and(pair_diff == 0, pair_var == 0)] = 1 # Handle 0/0 case
            y_pred = norm_s.cdf(pair_diff / np.sqrt(pair_var)) # Div/0 is fine
        elif self.P["preference_eqn"] == "bradley-terry":
            loss_bce, loss_0_1 = bt_loss_inner(
                # NOTE: Awkward torch <-> numpy conversion
                normalised_diff = torch.tensor(pair_diff / self._mean_ep_length, device=self.device).float(),
                y = y,
                equal_band=0.
            )
            return loss_bce if self.P["loss_func"] == "bce" else loss_0_1

    def n(self, tree, states, actions, next_states):
        assert len(states.shape) == 2
        n = torch.zeros(len(tree), device=self.device)
        for x in tree.get_leaf_nums(self.featuriser(states, actions, next_states).cpu().numpy()): n[x] += 1
        return n

# ==============================================================================
# UTILITIES

def least_squares_fitness(A, y, p_clip, preference_eqn):
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

def maximum_likelihood_fitness(A, y, preference_eqn, lr=0.1, epsilon=1e-5):
    """
    Construct maximum likelihood fitness estimates under the specified preference equation.
    Normalise fitness to be negative on the training set, with unit standard deviation.
    https://apps.dtic.mil/sti/pdfs/ADA543806.pdf.
    """
    f = norm.sample((A.shape[1],)).to(A.device) # Initialise with samples from standard normal
    f.requires_grad = True
    opt = torch.optim.Adam([f], lr=lr)
    loss = float("inf")
    while True:
        pair_diff = A @ f
        if preference_eqn == "thurstone": y_pred = norm.cdf(pair_diff)
        elif preference_eqn == "bradley-terry": y_pred = 1 / (1 + torch.exp(-pair_diff))
        new_loss = bce_loss(y_pred, y)
        new_loss.backward()
        opt.step()
        if torch.abs(new_loss - loss) < epsilon: break
        loss = new_loss; opt.zero_grad()
    return ((f - f.max()) / f.std()).detach().cpu().numpy(), new_loss
