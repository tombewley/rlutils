import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import torch
torch.set_printoptions(precision=3, edgeitems=10, linewidth=200, sci_mode=False)
from hyperrectangles.rules import *


class PetsExplainer:
    def __init__(self, agent, reward_model=None):
        self.agent, self.reward_model = agent, reward_model
        self.run_names = [] # Unused

    def per_timestep(self, ep, t, state, action, next_state, reward, done, info, extra):
        self.t = t

        # CEM review: visualise initial/final first-action distributions
        if True:
            first_actions = extra["actions"][:,:,0].squeeze()
            plt.figure(); ax = plt.axes(projection="3d")
            ax.scatter3D(*first_actions[0,:,:3].T, c="k")
            ax.scatter3D(*first_actions[-1,:,:3].T, c="g")
            ax.scatter3D(*action[:3], c="r", s=100)

        if False:
            features = self.reward_model.featuriser(
                       extra["states"][:,:,:-1,:], extra["actions"], extra["states"][:,:,1:,:])
            tree = self.reward_model.tree
            indices = torch.tensor(tree.get_leaf_nums(features.cpu().numpy(), one_hot=True), device=self.agent.device)

            from tufte.templates import stacked_bar_chart

            data = indices.sum(dim=(1,2)) # Plot over planning iterations
            # data = indices[0].sum(dim=(0)) # Plot over time for a given planning iteration
            _, ax = plt.subplots()
            stacked_bar_chart(ax, data.cpu().numpy(),
                            segment_colours=self.reward_model.r.cpu().numpy(),
                            sorted=True)


            if False:
                assert self.agent.P["gamma"] == 1., "Assumes undiscounted returns"
                returns = (indices.sum(axis=2) * self.reward_model.r).sum(axis=2)
                elites = returns.topk(self.agent.P["cem_elites"]).indices
                elites_mask = torch.zeros(returns.shape[:2], dtype=bool, device=self.agent.device)
                for i in range(len(elites_mask)): elites_mask[i,elites[i]] = True
                # Sweep through CEM planning iterations
                for ind,r,e in zip(indices, returns, elites_mask):
                    rdx, nz = compute_rdx(n_i= ind[e].float().mean(axis=0),
                                          n_j= ind[~e].float().mean(axis=0),
                                          r  = self.reward_model.r)

                    assert torch.isclose(rdx.sum(), (r[e].mean() - r[~e].mean()))
                    msx_pos, msx_neg = compute_msx(rdx.sum(dim=0))

                    print(rdx.sum(dim=0))
                    print(msx_pos, msx_neg)

                    # print(rdx[rdx > 0.].sum())

                    # ====================

                    # print(p_delta.sum(dim=0))
                    # print(self.reward_model.r)
                    # print(nz)
                    # print(rdx.sum())

                    # print(rule(self.reward_model.tree.leaves[-2]))

                    break

            # print(tree.siblings)
            # print(difference_rule(tree.leaves[0], tree.leaves[-2]))
            # print(rules(tree, pred_dims=["reward"]))

        plt.show()

    def per_episode(self, ep):

        # Rollout review: rollout the latest episode's action sequence from the initial state
        if False: 
            assert not self.agent.random_mode and len(self.agent.memory) >= self.t

            states, actions, _, _, next_states = self.agent.memory.sample(self.t, mode="latest", keep_terminal_next=True)
            if states is not None:
                states = cat([states, next_states[-1:]], dim=0) # Add final state.
                
                N = 20
                DIMS = (0, 2, 1)

                with no_grad(): sim_states, _, _ = self.agent.model.rollout(states[0].unsqueeze(0), 
                                                   actions=actions.expand(N,-1,-1).unsqueeze(2), ensemble_index="ts1_b")
                sim_states = sim_states.squeeze(2)

                plt.figure(); ax = plt.axes(projection="3d")
                for s in sim_states: ax.plot3D(*s[:,DIMS].T, c="gray", lw=0.5)
                ax.plot3D(*states[:,DIMS].T, c="k", lw=2)

        plt.show()
        self.t = 0
        return {}


def compute_rdx(n_i, n_j, r):
    """Compute the temporally-decomposed reward difference explanation (RDX)"""
    n_delta = n_i - n_j
    nz = n_delta.sum(dim=-2) != 0.
    return n_delta[...,nz] * r[nz], nz

def compute_msx(rdx):
    """Compute a minimal sufficient explanation (MSX+, MSX-) given an RDX vector"""
    d = -rdx[rdx < 0.].sum() # Disadvantage d = negated sum of negative elements
    rank = rdx.argsort()
    v = 0. # Just-insufficient advantage v = sum of all but last element in MSX+
    for i, x in enumerate(reversed(rank)):
        assert rdx[x] > 0., "Advantage must be positive."
        if (v + rdx[x]) > d: break
        v += rdx[x]
    msx_pos = rank[-(i+1):] # MSX+ = positive elements required to overcome d
    dd = 0.
    for i, x in enumerate(rank):
        if (dd - rdx[x]) > v: break
        dd -= rdx[x]
    msx_neg = rank[:i+1] # MSX- = negative elements required to overcome v
    return msx_pos, msx_neg
