import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import kendalltau
from .epic import epic_with_return


def epic(reward_functions, graph, num_canon=0, gamma=1.):
    """
    Equivalent-Policy Invariant Comparison (EPIC) pseudometric.
    """
    if num_canon == 0:
        canon_actions, canon_next_states = None, None
    else:
        # NOTE: Generate canonicalisation actions and next states by uniform subsampling
        rng = np.random.default_rng()
        all_actions, all_next_states = torch.cat(graph.actions), torch.cat(graph.next_states)
        canon_actions = all_actions[rng.choice(len(all_actions), size=num_canon, replace=False)]
        canon_next_states = all_next_states[rng.choice(len(all_next_states), size=num_canon, replace=False)]
    return epic_with_return(reward_functions,
           graph.states, graph.actions, graph.next_states, canon_actions, canon_next_states)

def preference_loss(reward_functions, graph, preference_eqn="bradley-terry", equal_band=0.):
    assert preference_eqn == "bradley-terry", "Thurstone not implemented"
    returns = predict_returns(reward_functions, graph)
    return_diff = returns[:,[i for i,_ in graph.edges]] - returns[:,[j for _,j in graph.edges]]
    return bt_loss_inner(
        normalised_diff = return_diff / np.mean(graph.ep_lengths), # NOTE: Normalise by mean ep length
        y = torch.tensor([d["preference"] for _,_,d in graph.edges(data=True)], device=graph.device),
        equal_band = equal_band
    )

def rank_correlation(reward_functions, graph):
    """
    Kendall's Tau-b rank correlation coefficient.
    """
    return squareform(pdist(predict_returns(reward_functions, graph),
           metric=lambda a, b: kendalltau(a, b).correlation)) + np.identity(len(reward_functions))

def predict_returns(reward_functions, graph):
    with torch.no_grad():
        return torch.stack([rewards_by_ep.sum(dim=1) for rewards_by_ep in torch.split(torch.stack([
               r(torch.vstack(graph.states), torch.vstack(graph.actions), torch.vstack(graph.next_states))
               for r in reward_functions]), graph.ep_lengths, dim=1)], dim=1)

def bt_loss_inner(normalised_diff, y, equal_band=0.):
    y_pred = 1 / (1 + torch.exp(-normalised_diff))
    # Binary cross-entropy loss
    loss_bce = torch.nn.BCELoss(reduction="none")(y_pred, y.expand(*y_pred.shape)).mean(dim=-1)
    # Modified 0-1 loss with a central band reserved for "equal" class
    y_shift, y_pred_shift = y - 0.5, y_pred - 0.5
    y_sign =      torch.sign(y_shift)      * (torch.abs(y_shift) > equal_band)
    y_pred_sign = torch.sign(y_pred_shift) * (torch.abs(y_pred_shift) > equal_band)
    loss_0_1 = torch.abs(y_sign - y_pred_sign).mean(dim=-1)
    assert not(torch.isnan(loss_bce).any()) and not(torch.isinf(loss_bce).any())
    assert not(torch.isnan(loss_0_1).any()) and not(torch.isinf(loss_0_1).any())
    return loss_bce, loss_0_1
