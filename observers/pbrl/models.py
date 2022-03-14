from ...common.networks import SequentialNetwork
from ...common.utils import reparameterise

import torch
import numpy as np
from scipy.special import xlogy, xlog1py
from scipy.stats import norm as norm_s
from tqdm import tqdm


norm = torch.distributions.Normal(0, 1)

bce_loss = torch.nn.BCELoss()
bce_loss_no_red = torch.nn.BCELoss(reduction="none")


# TODO: Split models out into separate files


class RewardNet:
    def __init__(self, device, feature_names, P):
        self.device = device
        self.P = P
        self.net = SequentialNetwork(device=self.device,
            input_shape=len(feature_names),
            # Mean and log standard deviation
            output_size=2, 
            # 3x 256 hidden units used in PEBBLE paper
            code=[(None, 256), "R", (256, 256), "R", (256, 256), "R", (256, None)], 
            # 3e-4 used in PEBBLE paper
            lr=3e-4, 
            )
        self.shift, self.scale = 0., 1.

    def __call__(self, features, normalise=True):
        mu, log_std = reparameterise(self.net(features), clamp=("soft", -2, 2), params=True)
        # NOTE: Scaling up std output helps avoid extreme probabilities
        mu, std = mu.squeeze(1), torch.exp(log_std).squeeze(1) * 100. 
        if normalise:            
            mu, std = (mu - self.shift) / self.scale, std / self.scale  
        var = torch.pow(std, 2.)
        return mu, var, std

    def fitness(self, features):
        mu, var, _ = self(features)
        return mu.sum(), var.sum()

    def update(self, _, features, ep_lengths, A, __, ___, y):
        ep_features = torch.split(features, ep_lengths)
        k_train, n_train = A.shape
        rng = np.random.default_rng()
        losses = []
        for _ in range(self.P["num_batches_per_update"]):
            batch = rng.choice(k_train, size=min(k_train, self.P["batch_size"]), replace=False)
            A_batch, y_batch = A[batch], y[batch]
            abs_A_batch = torch.abs(A_batch)
            in_batch = abs_A_batch.sum(axis=0) > 0
            F_pred, var_pred = torch.zeros(n_train, device=self.device), torch.zeros(n_train, device=self.device)
            for i in range(n_train):
                if in_batch[i]: F_pred[i], var_pred[i] = self.fitness(features=ep_features[i])
            if self.P["preference_eqn"] == "thurstone": 
                pair_diff = A_batch @ F_pred
                sigma = torch.sqrt(abs_A_batch @ var_pred)
                y_pred = norm.cdf(pair_diff / sigma)
                loss = bce_loss(y_pred, y_batch) # Binary cross-entropy loss, PEBBLE equation 4
            elif self.P["preference_eqn"] == "bradley-terry": 
                # https://github.com/rll-research/BPref/blob/f3ece2ecf04b5d11b276d9bbb19b8004c29429d1/reward_model.py#L142
                F_pairs = torch.vstack([F_pred[pair] for pair in abs_A_batch.bool()])
                log_y_pred = torch.nn.functional.log_softmax(F_pairs, dim=1)
                # NOTE: Relies on j coming first in the columns of A
                loss = -(torch.column_stack([1-y_batch, y_batch]) * log_y_pred).sum() / y_batch.shape[0]

                # y_pred = torch.exp(log_y_pred[:,1])
            # print(torch.vstack((y_pred, y_batch)).detach().numpy())

            self.net.optimise(loss, retain_graph=False)
            losses.append(loss.item())
            # print(losses[-1])
        # Normalise rewards to be negative on the training set, with unit standard deviation
        with torch.no_grad(): all_rewards, _, _ = self(features, normalise=False)
        self.shift, self.scale = all_rewards.max(), all_rewards.std()
        return {"preference_loss": np.mean(losses)}


class RewardTree:
    def __init__(self, device, feature_names, P):
        # === Lazy import ===
        import hyperrectangles as hr
        self.rules, self.diagram, self.rectangles, self.show_split_quality = \
        hr.rules, hr.diagram, hr.show_rectangles, hr.show_split_quality
        from hyperrectangles.utils import increment_mean_and_var_sum
        self.increment_mean_and_var_sum = increment_mean_and_var_sum
        # ===================
        self.device = device
        self.P = P
        space = hr.Space(dim_names=["ep", "reward"] + feature_names)
        root = hr.node.Node(space, sorted_indices=space.all_sorted_indices) 
        self.tree = hr.tree.Tree(
            name="reward_function", 
            root=root, 
            split_dims=space.idxify(feature_names), 
            eval_dims=space.idxify(["reward"]),
            )
        self.r = torch.zeros(self.m, device=self.device)
        self.var = torch.zeros(self.m, device=self.device) 
        self.history = {} # History of tree modifications

    @property
    def m(self): return len(self.tree.leaves)

    def __call__(self, features):
        indices = self.features_to_indices(features)
        mu, var = self.r[indices], self.var[indices]
        std = torch.sqrt(var)     
        return mu, var, std

    def fitness(self, features): 
        # https://www.statlect.com/probability-distributions/normal-distribution-linear-combinations
        n = self.n(features)
        return n @ self.r, n @ torch.diag(self.var) @ n.T

    def update(self, history_key, features, ep_lengths, A, i_list, j_list, y, reset_tree=True):
        """
        Update the tree-structured reward function given a preference dataset.
        If reset_tree=True, the tree is first pruned back to its root (i.e. start from scratch).
        """
        # Store private variables for this update and perform verification
        self._num_eps, self._i_list, self._j_list, self._y,  = len(ep_lengths), i_list, j_list, y.cpu().numpy()
        assert A.shape == (len(self._i_list), self._num_eps)
        assert len(features) == sum(ep_lengths)
        # Compute fitness estimates for connected episodes, and apply uniform temporal prior to obtain reward targets
        # NOTE: scaling by episode lengths (making ep fitness correspond to sum not mean) causes weird behaviour
        reward_target = fitness_case_v(A, y, self.P["p_clip"]) # * np.mean(ep_lengths) / ep_lengths
        # Populate tree. 
        self.tree.space.data = np.hstack((
            # NOTE: Using index in connected list rather than pbrl.episodes
            np.vstack([[[i, r]] * l for i, (r, l) in enumerate(zip(reward_target.cpu().numpy(), ep_lengths))]),
            features.cpu().numpy()                             
            ))
        if reset_tree: self.tree.prune_to(self.tree.root) 
        # NOTE: If we don't reiterate this on each update we get a weird bug of old self._attributes being referenced
        if not self.P["split_by_variance"]: self.tree.qual_func=self._split_qual_func
        self.tree.populate()
        num_samples = len(self.tree.space.data)
        # Store per-episode counts for each node of the tree
        assert reset_tree
        self.tree.root.counts = np.array(ep_lengths)
        self.tree.root.proxy_qual = {}
        # Calculate loss, pair_diff and pair_var for the initial state of the tree
        mean, var, counts = (np.array(attr) for attr in self.tree.gather(("mean","reward"), ("var","reward"), "counts"))
        self._current_loss, self._current_pair_diff, self._current_pair_var = self.preference_loss(mean, var, counts)
        history_split = [[self.m, self._current_loss, sum(self.tree.gather(("var_sum", "reward"))) / num_samples]]
        # Perform best-first splitting until m_max is reached
        with np.errstate(divide="ignore"): # __enter__ and __exit__ are both expensive here, so only do this once
            with tqdm(total=self.P["m_max"], initial=self.m, desc="Splitting") as pbar:
                while self.m < self.P["m_max"] and len(self.tree.split_queue) > 0:
                    node = self.tree.split_next_best(
                        min_samples_leaf=self.P["min_samples_leaf"], num_from_queue=self.P["num_from_queue"], store_all_qual=self.P["store_all_qual"])
                    if node is not None:
                        # Store counts in left and right children
                        node.left.counts = np.zeros_like(node.counts)
                        e, c = np.unique(np.rint(node.left.data(0)).astype(int), return_counts=True)
                        node.left.counts[e] = c
                        node.right.counts = node.counts - node.left.counts
                        node.left.proxy_qual, node.right.proxy_qual = {}, {}
                        # Calculate new loss, pair_diff and pair_var
                        mean, var, counts = (np.array(attr) for attr in self.tree.gather(("mean","reward"), ("var","reward"), "counts"))
                        new_loss, self._current_pair_diff, self._current_pair_var = self.preference_loss(mean, var, counts)
                        # if not self.P["split_by_variance"]: assert np.isclose(max(node.all_qual[node.split_dim]), self._current_loss - new_loss)
                        self._current_loss = new_loss
                        # NOTE: Empty split cache and recompute queue; necessary because split quality is not local
                        if not self.P["split_by_variance"]: self.tree._compute_split_queue()
                        # Append to history
                        history_split.append([self.m,
                            self._current_loss[0],                                     # Preference loss
                            sum(self.tree.gather(("var_sum", "reward"))) / num_samples # Proxy loss: variance
                            ])
                        pbar.update(1)
        # Wipe _current variables for safety; prevents further splitting except by calling this method
        self._current_loss, self._current_pair_diff, self._current_pair_var = None, None, None
        # Greedily prune one leaf at a time until tree is back to the root
        tree_before_prune = self.tree.clone()
        history_prune = [history_split[-1]]
        prune_indices = []
        mean, var, counts = (np.array(attr) for attr in self.tree.gather(("mean","reward"), ("var","reward"), "counts"))
        with tqdm(total=self.P["m_max"], initial=self.m, desc="Merging") as pbar:
            while self.m > 1:
                prune_candidates = []
                for i in range(self.m - 1):
                    left, right = self.tree.leaves[i:i+2]
                    if left.parent is right.parent:
                        m = np.delete(mean,   i, axis=0); m[i] = left.parent.mean[1] # NOTE: Assumes reward is dim 1
                        v = np.delete(var,    i, axis=0); v[i] = left.parent.cov[1,1]
                        c = np.delete(counts, i, axis=0); c[i] = left.parent.counts
                        loss, _, _ = self.preference_loss(m, v, c)
                        prune_candidates.append((i, m, v, c, loss))
                x, mean, var, counts, loss = sorted(prune_candidates, key=lambda cand: cand[4])[0]
                assert self.tree.prune_to(self.tree.leaves[x].parent) == {x, x+1}
                prune_indices.append(x)
                history_prune.append([self.m,
                    loss[0],                                                   # Preference loss
                    sum(self.tree.gather(("var_sum", "reward"))) / num_samples # Proxy loss: variance
                    ])
                pbar.update(1)
        # Restore the subtree in the pruning sequence with minimum size-regularised loss
        self.tree = tree_before_prune
        # NOTE: Using reversed list to ensure *last* occurrence returned
        optimum = (len(history_prune)-1) - np.argmin([l + (self.P["alpha"] * m) for m,l,_ in reversed(history_prune)])
        for x in prune_indices[:optimum]:
            assert self.tree.prune_to(self.tree.leaves[x].parent) == {x, x+1}
        # Log split/merge history and store component means and variances in PyTorch form
        self.history[history_key] = {"split": history_split, "prune": history_prune, "m": self.m}
        self.compute_r_and_var()
        self._num_eps, self._i_list, self._j_list, self._y = None, None, None, None
        print(self.tree.space)
        print(self.tree)
        print(self.rules(self.tree, pred_dims="reward", sf=5))#, out_name="tree_func"))
        return {"preference_loss": history_prune[optimum][1], "num_leaves": self.m}

    def _split_qual_func(self, node, split_dim, _, valid_split_indices): 
        """
        Evaluate the quality of all valid splits of node along split_dim.
        """
        ep_nums, rewards = node.space.data[node.sorted_indices[:,split_dim][:,None], [0,1]].T # NOTE: ep nums/rewards must be dims 0/1
        ep_nums = np.rint(ep_nums).astype(int)
        max_num_left = valid_split_indices[-1] + 1 # +1 needed
        mean = np.zeros((2, max_num_left))
        var_sum = mean.copy()
        mean[1,0] = node.mean[1] 
        var_sum[1,0] = node.var_sum[1]
        counts = np.zeros((2, max_num_left, self._num_eps), dtype=int)
        counts[1,0] = node.counts
        num_left_range = np.arange(max_num_left)
        num_range = np.stack((num_left_range, node.num_samples - num_left_range), axis=0)
        for num_left, num_right in num_range[:,1:].T:
            ep_num, reward = ep_nums[num_left-1], rewards[num_left-1]
            mean[0,num_left], var_sum[0,num_left] = self.increment_mean_and_var_sum(num_left,  mean[0,num_left-1], var_sum[0,num_left-1], reward, 1)
            mean[1,num_left], var_sum[1,num_left] = self.increment_mean_and_var_sum(num_right, mean[1,num_left-1], var_sum[1,num_left-1], reward, -1)                    
            counts[:,num_left] = counts[:,num_left-1] # Copy
            counts[0,num_left,ep_num] = counts[0,num_left-1,ep_num] + 1 # Update
            counts[1,num_left,ep_num] = counts[1,num_left-1,ep_num] - 1 
        node.proxy_qual[split_dim] = var_sum[1,0] - var_sum[:,valid_split_indices].sum(axis=0)
        num_range[0,0] = 1 # Prevent div/0 warning
        loss, _, _ = self.preference_loss(mean, var_sum / num_range, counts, split_mode=True)
        return self._current_loss - loss[valid_split_indices]

    def preference_loss(self, mean, var, counts, split_mode=False):
        """
        Compute preference loss given vectors of per-component means and variances,
        and a matrix of counts for each episode-component pair. In split mode, these arrays each
        have an additional dimension (all possible split locations) but only two rows (left and right),
        and we need to add self._current_pair_diff and self._current_pair_var to compute the global loss.
        """
        assert mean.shape[0] == var.shape[0] == counts.shape[0]
        if split_mode: 
            assert len(mean.shape) == len(var.shape) == 2 and len(counts.shape) == 3 and counts.shape[0] == 2
        else:
            assert len(mean.shape) == len(var.shape) == 1 and len(counts.shape) == 2
            mean, var, counts = np.expand_dims(mean, 1), np.expand_dims(var, 1), np.expand_dims(counts, 1)
        i_counts = counts[:,:,self._i_list]
        j_counts = counts[:,:,self._j_list]
        pair_diff = (np.expand_dims(mean, 2) * (i_counts - j_counts)).sum(axis=0)
        pair_var = (np.expand_dims(var, 2) * (i_counts**2 + j_counts**2)).sum(axis=0)
        if split_mode: # Need to add pair_diff and pair_var contributions from the rest of the tree
            assert self._current_pair_diff.shape == self._current_pair_var.shape == pair_diff[0].shape == pair_var[0].shape
            # pair_diff[0] and pair_var[0] are contributions of this node to totals pre-splitting   
            pair_diff += self._current_pair_diff - pair_diff[0]
            pair_var = np.maximum(pair_var + self._current_pair_var - pair_var[0], 0) # Clip at zero
        pair_var[np.logical_and(pair_diff == 0, pair_var == 0)] = 1 # Handle 0/0 case
        y_pred = norm_s.cdf(pair_diff / np.sqrt(pair_var)) # Div/0 is fine
        # Robust cross-entropy loss (https://stackoverflow.com/a/50024648)
        loss = (-(xlogy(self._y, y_pred) + xlog1py(1 - self._y, -y_pred))).mean(axis=1)
        assert not np.isnan(np.einsum("i->", loss))
        return loss, pair_diff[0], pair_var[0]
        
    def features_to_indices(self, features):
        return [self.tree.leaves.index(next(iter(
                self.tree.propagate([None,None]+list(f), mode="max")))) for f in features.cpu().numpy()]

    def n(self, features):
        n = torch.zeros(self.m, device=self.device)
        for x in self.features_to_indices(features): n[x] += 1
        return n

    def compute_r_and_var(self):
        self.r = torch.tensor(self.tree.gather(("mean","reward")), device=self.device).float()
        self.var = torch.tensor(self.tree.gather(("var","reward")), device=self.device).float()

# ==============================================================================
# UTILITIES

def fitness_case_v(A, y, p_clip):
    """
    Construct fitness estimates under Thurstone's Case V model. 
    Uses Morrissey-Gulliksen least squares for incomplete comparison matrix.
    """
    d = norm.icdf(torch.clamp(y, p_clip, 1-p_clip)) # Clip to prevent infinite values
    f, _, _, _ = torch.linalg.lstsq(A.T @ A, A.T @ d, rcond=None)
    return f - f.max() # NOTE: Shift so that maximum fitness is zero (cost function)
