import torch
from numpy import unravel_index
from numpy.random import default_rng


norm = torch.distributions.Normal(0, 1)


class Sampler:
    def __init__(self, pbrl, P): 
        self.pbrl, self.P = pbrl, P
        self.seed()

    def seed(self, seed=None):
        self.pt_rng = torch.Generator(device=self.pbrl.device)
        if seed is not None: self.pt_rng.manual_seed(seed)
        self.np_rng = default_rng(seed)

    def __iter__(self): 
        """
        Precompute weighting matrix for the current batch.
        """
        self._k = 0
        if self.P["weight"] == "uniform":
            n = len(self.pbrl.graph); self.w = torch.zeros((n, n), device=self.pbrl.device)
        else:
            with torch.no_grad(): 
                mu, var = torch.tensor([self.pbrl.fitness(ep["transitions"]) for _, ep in self.pbrl.graph.nodes(data=True)], device=self.pbrl.device).T
            if "ucb" in self.P["weight"]:
                self.w = ucb_sum(mu, var, num_std=self.P["num_std"])
                if self.P["weight"] == "ucb_r": self.w = -self.w # Invert
            elif self.P["weight"] == "entropy": 
                self.w = preference_entropy(mu, var, preference_eqn=self.P["preference_eqn"])
        return self

    def __next__(self):
        """
        Sample a trajectory pair from the current weighting matrix subject to constraints.
        """
        if self._k >= self.batch_size: return 1, None, None, None # Batch completed
        n = len(self.pbrl.graph); assert self.w.shape == (n, n)
        not_rated = torch.isnan(self.pbrl.graph.preference_matrix) # TODO: Can bypass this and compute directly from graph
        if not_rated.sum() <= n: return 2, None, None, None # Fully connected
        p = self.w.clone()
        # Enforce non-identity constraint...
        p.fill_diagonal_(float("nan"))
        # ...enforce non-repeat constraint...
        rated = ~not_rated
        p[rated] = float("nan")
        # ...enforce connectedness constraint...
        unconnected = rated.sum(axis=1) == 0
        if sum(unconnected) < n: p[unconnected] = float("nan") # (ignore connectedness if first ever rating)
        if self.P["recency_constraint"]:
            # ...enforce recency constraint...
            p[:self.ij_min, :self.ij_min] = float("nan")
        nans = torch.isnan(p)
        if self.P["probabilistic"]: # NOTE: Approach used in AAMAS paper
            # ...rescale into a probability distribution...
            p -= torch.min(p[~nans]) 
            if torch.nansum(p) == 0: p[~nans] = 1
            p[nans] = 0
            # ...and sample a pair from the distribution
            i, j = unravel_index(list(torch.utils.data.WeightedRandomSampler(
                   weights=p.ravel(), num_samples=1, generator=self.pt_rng))[0], p.shape)
        else: 
            # ...and pick at random from the set of argmax pairs
            raise NotImplementedError("Avoid use of argwhere")
            argmaxes = argwhere(p == torch.max(p[~nans])).T
            i, j = argmaxes[self.np_rng.choice(len(argmaxes))]; i, j = i.item(), j.item()
        # Check that all constraints are satisfied
        assert i != j and not_rated[i, j]
        if self.P["recency_constraint"]:
            if sum(unconnected) < n: assert rated[i].sum() > 0
            assert i >= self.ij_min or j >= self.ij_min
        self._k += 1
        print(self.w)
        print(p)
        return 0, i, j, p

def ucb_sum(mu, var, num_std):
    ucb = mu + num_std * torch.sqrt(var)
    return ucb.reshape(-1,1) + ucb.reshape(1,-1)

def preference_entropy(mu, var, preference_eqn): # TODO: Redundancy with models.py
    F_diff = mu.reshape(-1,1) - mu.reshape(1,-1)
    if preference_eqn == "thurstone": 
        sigma = torch.sqrt(var.reshape(-1,1) + var.reshape(1,-1))
        sigma[torch.logical_and(F_diff == 0, sigma == 0)] = 1 # Handle 0/0 case
        y_pred = norm.cdf(F_diff / sigma)
    elif preference_eqn == "bradley-terry": 
        raise NotImplementedError()
    y_log_y = torch.nan_to_num(y_pred * torch.log(y_pred), 0)
    return -(y_log_y + y_log_y.T)
