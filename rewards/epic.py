from torch import no_grad, vstack, stack, sqrt, corrcoef
import networkx as nx
from sklearn.manifold import MDS


def epic(reward_functions, states, actions, next_states, canon_actions, canon_next_states, gamma=1.):
    """
    Equivalent-Policy Invariant Comparison (EPIC) pseudometric. From:
        Gleave, Adam et al. "Quantifying Differences in Reward Functions." ICLR 2021.
    """
    # Compute canonicalised rewards, which are invariant to potential-based shaping
    n_v = len(states)
    assert len(actions) == len(next_states) == n_v
    with no_grad():
        rewards = stack([r(states, actions, next_states) for r in reward_functions])
        means = stack([mean_rewards(r, vstack((states, next_states)), canon_actions, canon_next_states) for r in reward_functions])
        canon_rewards = rewards + gamma * means[:,n_v:] - means[:,:n_v]
    # Compute all pairwise Pearson distances and represent as an undirected networkx graph
    return nx.from_numpy_matrix(sqrt(0.5 * (1 - corrcoef(canon_rewards))).cpu().numpy())

def mean_rewards(reward_function, states, actions, next_states):
    """
    Given n_v states, n_m actions and n_m next states, compute the n_v x n_m array
    of rewards for all permutations, then take the mean along the second dimension.
    """
    n_v, n_m = len(states), len(next_states)
    assert len(actions) == n_m
    rewards = reward_function(states.unsqueeze(1).expand(n_v, n_m, states.shape[1]),
                              actions.unsqueeze(0).expand(n_v, n_m, actions.shape[1]),
                              next_states.unsqueeze(0).expand(n_v, n_m, next_states.shape[1]))
    return rewards.mean(dim=1)

def mds_layout(g):
    """
    Networkx graph layout using scikit-learn's multidimensional scaling tool
    """
    return MDS(max_iter=3000, eps=1e-9, dissimilarity="precomputed", n_init=10
              ).fit(nx.to_numpy_matrix(g)).embedding_


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
