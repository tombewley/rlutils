from torch import no_grad, vstack, stack, split, sqrt, corrcoef
import networkx as nx
from sklearn.manifold import MDS


def epic(reward_functions, states, actions, next_states, canon_actions=None, canon_next_states=None, gamma=1.):
    """
    Equivalent-Policy Invariant Comparison (EPIC) pseudometric. From:
        Gleave, Adam et al. "Quantifying Differences in Reward Functions." ICLR 2021.
    NOTE: Returns Pearson correlations rather than distances
    """
    n_v = len(states)
    assert len(actions) == len(next_states) == n_v
    with no_grad():
        rewards = stack([r(states, actions, next_states) for r in reward_functions])
        if canon_actions is not None:
            # Compute canonicalised rewards, which are invariant to potential-based shaping
            means = stack([mean_rewards(r, vstack((states, next_states)), canon_actions, canon_next_states) for r in reward_functions])
            rewards += gamma * means[:,n_v:] - means[:,:n_v]
    # Compute all pairwise Pearson correlations
    return corrcoef(rewards), rewards

def epic_with_return(reward_functions, states_by_ep, actions_by_ep, next_states_by_ep, canon_actions=None, canon_next_states=None, gamma=1.):
    corr_r, rewards = epic(reward_functions, vstack(states_by_ep), vstack(actions_by_ep), vstack(next_states_by_ep), canon_actions, canon_next_states, gamma)
    returns = stack([rewards_by_ep.sum(dim=1) for rewards_by_ep in split(rewards, [len(s) for s in states_by_ep], dim=1)], dim=1)
    return corr_r, corrcoef(returns), rewards, returns 

def mean_rewards(reward_function, states, actions, next_states):
    """
    Given n_v states, n_m actions and n_m next states, compute the n_v x n_m array
    of rewards for all permutations, then take the mean along the second dimension
    """
    n_v, n_m = len(states), len(next_states)
    assert len(actions) == n_m
    rewards = reward_function(states.unsqueeze(1).expand(n_v, n_m, states.shape[1]),
                              actions.unsqueeze(0).expand(n_v, n_m, actions.shape[1]),
                              next_states.unsqueeze(0).expand(n_v, n_m, next_states.shape[1]))
    return rewards.mean(dim=1)

def graph(corr):
    """Represent Pearson correlation matrix as an undirected networkx graph"""
    return nx.from_numpy_matrix(corr.cpu().numpy())

def mds_layout(g):
    """
    Networkx graph layout using scikit-learn's multidimensional scaling tool
    NOTE: Convert Pearson correlations to distances
    """
    return MDS(max_iter=3000, eps=1e-9, dissimilarity="precomputed", n_init=10
              ).fit(corr_to_dist(nx.to_numpy_matrix(g))).embedding_

def corr_to_dist(corr):
    return sqrt(0.5 * (1 - corr))

if __name__ == "__main__":
    from torch import rand
    import matplotlib.pyplot as plt

    def r0(states, actions, next_states):
        return ((next_states[...,0] - states[...,0]) > -0.5).float()
    def r1(states, actions, next_states):
        return ((next_states[...,0] - states[...,0]) > 0.).float()
    def r2(states, actions, next_states):
        return ((next_states[...,0] - states[...,0]) > 0.5).float()
    def r3(states, actions, next_states):
        return ((next_states[...,0] - states[...,0]) < -0.5).float()
    def r4(states, actions, next_states):
        return ((next_states[...,0] - states[...,0]) < 0.).float()
    def r5(states, actions, next_states):
        return ((next_states[...,0] - states[...,0]) < 0.01).float()

    n_v = 100
    n_m = 50

    g = epic([r0,r1,r2,r3,r4,r5], rand(n_v, 3), rand(n_v, 2), rand(n_v, 3), rand(n_m, 2), rand(n_m, 3))
    nx.draw(g, pos=mds_layout(g), with_labels=True, edgelist=[(4,5)])
    plt.show()
