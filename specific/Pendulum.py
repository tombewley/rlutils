import torch


def reward_function(states, actions, next_states):
    """Reward function for Pendulum-v0."""
    def angle_normalize(x): 
        return ((x + torch.pi) % (2 * torch.pi)) - torch.pi
    th = torch.arccos(torch.clamp(states[:,0], -1, 1))
    thdot = states[:,2]
    u = torch.clamp(actions, -2, 2).squeeze()
    costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)
    return -costs
