import torch
from numpy import unravel_index, argwhere
from numpy.random import choice


class Sampler:
    def __init__(self, pbrl, P): self.pbrl, self.P = pbrl, P

    def __iter__(self): 
        """
        Precompute weighting matrix for the current batch.
        """
        self.k = 0
        if self.P["weight"] == "uniform":
            n = len(self.pbrl.episodes); self.w = torch.zeros((n, n), device=self.device)
        else:
            with torch.no_grad(): 
                mu, var = torch.tensor([self.pbrl.fitness(ep) for ep in self.pbrl.episodes], device=self.pbrl.device).T
            if "ucb" in self.P["weight"]:
                self.w = ucb_sum(mu, var, num_std=self.P["num_std"])
                if self.P["weight"] == "ucb_r": self.w = -self.w # Invert
            elif self.P["weight"] == "uncertainty": 
                raise NotImplementedError()
        return self

    def __next__(self):
        """
        Sample a trajectory pair from the current weighting matrix subject to constraints.
        """
        print(self.k)
        if self.k >= self.batch_size: return 1, None, None, None # Batch completed
        n = self.pbrl.Pr.shape[0]; assert self.w.shape == (n, n)
        not_rated = torch.isnan(self.pbrl.Pr)
        if not_rated.sum() <= n: return 2, None, None, None # Fully connected
        p = self.w.clone()
        if not self.P["constrained"]: raise NotImplementedError()
        # Enforce non-identity constraint...
        p.fill_diagonal_(float("nan"))
        # ...enforce non-repeat constraint...
        rated = ~not_rated
        p[rated] = float("nan")
        # ...enforce connectedness constraint...    
        unconnected = argwhere(rated.sum(axis=1) == 0).flatten()
        if len(unconnected) < n: p[unconnected] = float("nan") # (ignore connectedness if first ever rating)
        # ...enforce recency constraint...
        p[:self.ij_min, :self.ij_min] = float("nan")
        nans = torch.isnan(p)
        if self.P["probabilistic"]: # NOTE: Approach used in AAMAS paper
            # ...rescale into a probability distribution...
            p -= torch.min(p[~nans]) 
            if torch.nansum(p) == 0: p[~nans] = 1
            p[nans] = 0
            # ...and sample a pair from the distribution
            i, j = unravel_index(list(torch.utils.data.WeightedRandomSampler(p.ravel(), num_samples=1))[0], p.shape)
        else: 
            # ...and pick at random from the set of argmax pairs
            argmaxes = argwhere(p == torch.max(p[~nans])).T
            i, j = argmaxes[choice(len(argmaxes))]; i, j = i.item(), j.item()
        # Check that all constraints are satisfied
        assert i != j and not_rated[i, j] and (i >= self.ij_min or j >= self.ij_min)
        if len(unconnected) < n: assert rated[i].sum() > 0 
        self.k += 1
        return 0, i, j, p


def ucb_sum(mu, var, num_std):
    ucb = mu + num_std * torch.sqrt(var)
    return ucb.reshape(-1,1) + ucb.reshape(1,-1)