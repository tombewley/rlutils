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
# TODO: Generic model for others to inherit from


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
            # print(loss.item())

            self.net.optimise(loss, retain_graph=False)

            # from torchviz import make_dot
            # make_dot(loss).render("loss_graph", format="png")
        
        # Normalise rewards to be negative on the training set, with unit standard deviation
        with torch.no_grad(): all_rewards, _, _ = self(features, normalise=False)
        self.shift, self.scale = all_rewards.max(), all_rewards.std()

        # all_rewards = self.reward(features=features).detach().numpy()
        # print(all_rewards.max(), all_rewards.std())
        # plt.hist(all_rewards)
        # plt.show()


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
            qual_func=self._split_qual_func
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
        If reset_tree=True, tree is first pruned back to its root (i.e. start from scratch).
        """
        # Store private variables for this update and perform verification
        self._i_list, self._j_list, self._y = i_list, j_list, y
        self._num_eps = len(ep_lengths)
        assert A.shape == (len(self._i_list), self._num_eps)
        assert len(features) == sum(ep_lengths)
        # Compute fitness estimates for connected episodes, and apply uniform temporal prior to obtain reward targets
        # NOTE: scaling by episode lengths (making ep fitness correspond to sum not mean) causes weird behaviour
        reward_target = fitness_case_v(A, y, self.P["p_clip"]) # * np.mean(ep_lengths) / ep_lengths
        # Populate tree. 
        self.tree.space.data = np.hstack((
            # NOTE: Using index in connected list rather than pbrl.episodes
            np.vstack([[[i, r]] * l for i, (r, l) in enumerate(zip(reward_target.cpu().numpy(), ep_lengths))]),
            # np.vstack([[[i, r]] * l for (i, r, l) in zip(ep_nums, reward_target.cpu().numpy(), ep_lengths)]),
            features.cpu().numpy()                             
            ))
        if reset_tree: self.tree.prune_to(self.tree.root) 
        self.tree.populate()
        num_samples = len(self.tree.space.data)
        # Store per-episode counts for each node of the tree
        assert reset_tree
        self.tree.root.counts = np.array(ep_lengths)
        self.tree.root.proxy_qual = {}
        # Calculate loss, pair_diff and pair_var for the initial state of the tree
        mean, var, counts = self.tree.gather(("mean","reward"), ("var","reward"), "counts")
        self._current_loss, self._current_pair_diff, self._current_pair_var = self.preference_loss(np.array(mean), np.array(var), np.array(counts))
        history_split = [[self.m, self._current_loss, sum(self.tree.gather(("var_sum", "reward"))) / num_samples]]
        # Perform best-first splitting until m_max is reached
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
                    mean, var, counts = self.tree.gather(("mean","reward"), ("var","reward"), "counts")
                    new_loss, self._current_pair_diff, self._current_pair_var = self.preference_loss(np.array(mean), np.array(var), np.array(counts))
                    # if self.P["loss"] == "preference": assert np.isclose(max(node.all_qual[node.split_dim]), self._current_loss - new_loss)
                    self._current_loss = new_loss
                    # Append to history
                    history_split.append([self.m,
                        self._current_loss,                                        # Preference loss
                        sum(self.tree.gather(("var_sum", "reward"))) / num_samples # Proxy loss: variance
                        ])
                    # NOTE: Empty split cache and recompute queue; necessary because split quality is not local
                    self.tree._compute_split_queue()
                    pbar.update(1)

        """
        TODO: Pruning
        Linearity of MCCP does *not* hold for preference loss, so cannot rely on weakest link pruning
        Instead, be more conservative and only prune one leaf at a time, 
        greedily picking the one that yields the smallest reduction in preference loss
        """
        
        history_merge = [history_split[-1]]
        if False:
            # Perform minimal cost complexity pruning until labelling loss is minimised.
            ep_features = torch.split(features, ep_lengths)
            N = torch.vstack([self.n(f) for f in ep_features])
            tree_before_merge = self.tree.clone()                
            parent_num, pruned_nums = None, None
            with tqdm(total=self.P["m_max"], initial=self.m, desc="Merging") as pbar:
                while True: 
                    # Measure loss.
                    r, var, var_sum = self.tree.gather(("mean","reward"),("var","reward"),("var_sum","reward"))
                    r, var = torch.tensor(r, device=self.device).float(), torch.tensor(var, device=self.device).float()
                    history_merge.append([self.m, parent_num, pruned_nums,
                        labelling_loss(A, y, N, r, var, self.P["p_clip"]).item(), # True labelling loss.
                        sum(var_sum) / num_samples # Proxy loss: variance.
                        ])
                    if self.m <= self.P["m_stop_merge"]: break
                    # Perform prune.
                    parent_num, pruned_nums = self.tree.prune_mccp()
                    pbar.update(-1)
                    # Efficiently update N array (NOTE: relies on pruned_nums being consecutive and ordered)
                    assert pruned_nums[-1] == pruned_nums[0] + len(pruned_nums)-1
                    N[:,pruned_nums[0]] = N[:,pruned_nums].sum(axis=1)
                    N = torch.cat((N[:,:pruned_nums[1]], N[:,pruned_nums[-1]+1:]), axis=1)
                    if False: assert (N == torch.vstack([self.n(f) for f in ep_features])).all() # Sense check.     
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

        # Delete _current variables for safety; prevents further splitting except by calling this method
        del self._current_loss, self._current_pair_diff, self._current_pair_var

        self.compute_r_and_var()
        print(self.tree.space)
        print(self.tree)
        print(self.rules(self.tree, pred_dims="reward", sf=5))#, out_name="tree_func"))

    def _split_qual_func(self, node, split_dim, _, valid_split_indices): 
        """
        NOTE: Assumes ep nums and rewards are in dimensions 0 and 1 of node.space.data respectively.
        """
        ep_nums, rewards = node.space.data[node.sorted_indices[:,split_dim][:,None], [0,1]].T 
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
        
        assert (counts.sum(axis=0) == counts[1,0]).all()
        num_range[0,0] = 1 # Prevent div/0 warning
        loss, _, _ = self.preference_loss(mean, var_sum / num_range, counts, split_mode=True)

        node.proxy_qual[split_dim] = var_sum[1,0] - var_sum[:,valid_split_indices].sum(axis=0)
        if self.P["loss"] == "variance":     return node.proxy_qual[split_dim]
        elif self.P["loss"] == "preference": return self._current_loss - loss[valid_split_indices]

    def preference_loss(self, mean, var, counts, split_mode=False):
        """
        xxx
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
        with np.errstate(divide="ignore"):
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

def labelling_loss(A, y, N, r, var, p_clip, old=False):
    """
    Loss function l that this algorithm is ultimately trying to minimise.
    """
    N_diff = A @ N
    pair_diff = N_diff @ r
    if old: sigma = torch.sqrt(N_diff**2 @ var) # Faster than N A^T diag(var) A N^T
    else:   sigma = torch.sqrt(torch.abs(A) @ N**2 @ var)   
    sigma[torch.logical_and(pair_diff == 0, sigma == 0)] = 1 # Handle 0/0 case
    y_pred = norm.cdf(pair_diff / sigma) # Div/0 is fine
    if old:
        d = norm.ppf(torch.clamp(y, p_clip, 1-p_clip)) 
        d_pred = norm.ppf(torch.clamp(y_pred, p_clip, 1-p_clip)) 
        assert not np.isnan(d_pred).any()
        return ((d_pred - d)**2).mean() # MSE loss
    else:
        return bce_loss(y_pred, y) # BCE loss

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