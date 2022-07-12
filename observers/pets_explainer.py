import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from torch import cat, no_grad


class PetsExplainer:
    def __init__(self, agent, reward_model=None):
        self.agent, self.reward_model = agent, reward_model
        self.run_names = [] # Unused

    def per_timestep(self, ep, t, state, action, next_state, reward, done, info, extra):
        self.t = t

        # CEM review: visualise initial/final first-action distributions
        if False:
            first_actions = extra["actions"][:,:,0].squeeze()
            plt.figure(); ax = plt.axes(projection="3d")
            ax.scatter3D(*first_actions[0,:,:3].T, c="k")
            ax.scatter3D(*first_actions[-1,:,:3].T, c="g")
            ax.scatter3D(*action[:3], c="r", s=100)

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
