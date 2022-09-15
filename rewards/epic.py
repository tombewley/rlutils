from torch import no_grad, split, stack, hstack, vstack, corrcoef


def epic(states, actions, next_states, reward_functions=None, rewards=None, canon_actions=None, canon_next_states=None, gamma=1.):
    """
    Equivalent-Policy Invariant Comparison (EPIC) pseudometric. From:
        Gleave, Adam et al. "Quantifying Differences in Reward Functions." ICLR 2021.
    NOTE: Returns Pearson correlations rather than distances
    """
    n_v = len(states)
    assert len(actions) == len(next_states) == n_v
    with no_grad():
        if rewards is None: rewards = stack([r(states, actions, next_states) for r in reward_functions])
        if canon_actions is not None:
            # Compute canonicalised rewards, which are invariant to potential-based shaping
            means = stack([mean_rewards(r, vstack((states, next_states)), canon_actions, canon_next_states) for r in reward_functions])
            rewards += gamma * means[:,n_v:] - means[:,:n_v]
    # Compute all pairwise Pearson correlations
    return corrcoef(rewards), rewards

def epic_with_return(states_by_ep, actions_by_ep, next_states_by_ep,
                     reward_functions=None, rewards_by_ep=None, canon_actions=None, canon_next_states=None, gamma=1.):
    corr_r, rewards = epic(vstack(states_by_ep), vstack(actions_by_ep), vstack(next_states_by_ep),
                           reward_functions, None if rewards_by_ep is None else hstack(rewards_by_ep),
                           canon_actions, canon_next_states, gamma)
    if rewards_by_ep is None: rewards_by_ep = split(rewards, [len(s) for s in states_by_ep], dim=1)
    returns = stack([r.sum(dim=1) for r in rewards_by_ep], dim=1)
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

if __name__ == "__main__":
    from torch import rand
    import matplotlib.pyplot as plt
    from rlutils.rewards.evaluate import corr_to_dist, graph, draw_graph

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

    corr_r, _ = epic(states=rand(n_v, 3), actions=rand(n_v, 2), next_states=rand(n_v, 3),
                reward_functions=[r0,r1,r2,r3,r4,r5], canon_actions=rand(n_m, 2), canon_next_states=rand(n_m, 3))
    draw_graph(graph(corr_to_dist(corr_r)), node_size=0, with_labels=True, edgelist=[(4,5)])
    plt.show()
