from ...rewards.pbrl.graph import PreferenceGraph
from ...rewards.pbrl.sampler import Sampler
from ...rewards.pbrl.interactions import preference_batch
from ...rewards.epic import epic, epic_with_return
from .explainer import Explainer

import os
from torch import save as pt_save, load as pt_load, device, cuda


class PbrlObserver:
    """
    xxx
    """
    def __init__(self, P, run_names=None):
        self.P = {} if P is None else P
        self.run_names = run_names if run_names is not None else []
        # Preference graph, reward model, trajectory pair sampler, preference collection interface and explainer are all modular
        self.graph = PreferenceGraph()
        self.model = self.P["model"]["class"](self.P["model"]) if "model" in self.P else None
        self.sampler = Sampler(self.graph, self.model, self.P["sampler"]) if "sampler" in self.P else None
        self.interface = self.P["interface"]["class"](self.graph, self.P["interface"]) if "interface" in self.P else None
        self.explainer = Explainer(self, self.P["explainer"] if "explainer" in self.P else {})
        self._observing = "observe_freq" in self.P and self.P["observe_freq"] > 0
        self._saving = "save_freq" in self.P and self.P["save_freq"] > 0
        self._online = "feedback_freq" in self.P and self.P["feedback_freq"] > 0
        if self._online:
            assert self.model is not None and self.sampler is not None and self.interface is not None
            assert self._observing
            assert self.P["feedback_freq"] % self.P["observe_freq"] == 0    
            b = self.P["num_episodes_before_freeze"] / self.P["feedback_freq"]
            assert b % 1 == 0
            self._num_batches = int(b)
            self._batch_num = 0
            self._n_on_prev_batch = 0
        self._current_ep = []
        self._cum_rew_sum_or = 0.
    
    def link(self, agent):
        """
        Link the reward model to the replay memory of an off-policy RL agent.
        """
        assert len(agent.memory) == 0, "Agent must be at the start of learning."
        assert agent.device == self.graph.device
        agent.memory.__init__(agent.memory.capacity, reward=self.reward, relabel_mode="eager")
        if not agent.memory.lazy_reward: self.relabel_memory = agent.memory.relabel

    def reward(self, states, actions, next_states):
        assert self.P["reward_source"] != "extrinsic", "This shouldn't have been called. Unwanted call to pbrl.link(agent)?"
        if self.P["reward_source"] == "oracle": return self.interface.oracle(states, actions, next_states)
        if self.P["reward_source"] == "model": return self.model(states, actions, next_states) 

    def per_timestep(self, ep_num, t, state, action, next_state, reward, done, info, extra):
        self._current_ep.append((state, action, next_state))
            
    def per_episode(self, ep_num):
        """
        Operations to complete at the end of an episode, which may include adding self._current_ep
        to the preference graph, creating logs, and (if self._online==True), occasionally gathering
        a preference batch and updating the reward model.
        """   
        s, a, ns = self.graph.tensorise(self._current_ep)
        self._current_ep = []
        logs = {}
        # Log reward sums
        if self.P["reward_source"] == "model": 
            logs["reward_sum_model"] = self.model.fitness(s, a, ns)[0].item()
        if self.interface is not None and self.interface.oracle is not None: 
            logs["reward_sum_oracle"] = sum(self.interface.oracle(s, a, ns)).item()
            self._cum_rew_sum_or += logs["reward_sum_oracle"]
            logs["cumulative_reward_sum_oracle"] = self._cum_rew_sum_or
        # Add episodes to the preference graph with a specified frequency
        if self._observing and (ep_num+1) % self.P["observe_freq"] == 0:
            self.graph.add_episode(s, a, ns, run_name=self.run_names[-1], ep_num=ep_num)
        if self._online:
            if (ep_num+1) % self.P["feedback_freq"] == 0 and (ep_num+1) <= self.P["num_episodes_before_freeze"]:
                # Calculate batch size.
                if self.P["scheduling_coef"] > 0: # TODO: Make adaptive to remaining budget
                    assert self.sampler.P["recency_constraint"]
                    K = self.P["feedback_budget"]
                    B = self._num_batches
                    f = self.P["feedback_freq"] / self.P["observe_freq"] # Number of episodes between batches
                    c = self.P["scheduling_coef"]
                    b = self._batch_num # Current batch number.
                    batch_size = int(round((K / B * (1 - c)) + (K * (f * (2*(b+1) - 1) - 1) / (B * (B*f - 1)) * c)))
                else:
                    K = self.P["feedback_budget"] - len(self.graph.edges) # Remaining budget
                    B = self._num_batches - self._batch_num # Remaining number of batches
                    batch_size = int(round(K / B))
                # Gather preferences and update reward model
                logs.update(preference_batch(
                    sampler=self.sampler,
                    interface=self.interface,
                    graph=self.graph,
                    batch_size=batch_size,
                    ij_min=self._n_on_prev_batch,
                    history_key=(ep_num+1),
                    budget=self.P["feedback_budget"]
                ))
                logs.update(self.model.update(
                    graph=self.graph,
                    history_key=(ep_num+1)
                ))
                # If using oracle, measure alignment
                if self.interface.oracle is not None:
                    corr_r, corr_g, _, _ = epic_with_return([self.interface.oracle, self.model], self.graph.states, self.graph.actions, self.graph.next_states)
                    logs["reward_correlation"], logs["return_correlation"] = corr_r[0,1], corr_g[0,1]
                self.relabel_memory() # If applicable, relabel the agent's replay memory using the updated reward
                self._batch_num += 1 
                self._n_on_prev_batch = len(self.graph)
            # Periodically log and save out
            if self.explainer.P and (ep_num+1) % self.explainer.P["freq"] == 0: self.explainer(history_key=(ep_num+1))
        if self._saving and (ep_num+1) % self.P["save_freq"] == 0: self.save(history_key=(ep_num+1))
        return logs

    def relabel_memory(self): pass

    def save(self, history_key):
        path = f"{self.P['save_path']}/{self.run_names[-1]}"
        if not os.path.exists(path): os.makedirs(path)
        pt_save({
            "graph": self.graph,
            "model": self.model
        }, f"{path}/{history_key}.pbrl")

def load(fname, P=None):
    """
    Make an instance of PbrlObserver from the information stored by the .save() method.
    """
    device_ = device("cuda" if cuda.is_available() else "cpu")
    dict = pt_load(fname, device_)
    pbrl = PbrlObserver(P)
    pbrl.graph = dict["graph"]
    pbrl.model = dict["model"]
    pbrl.graph.device = device_
    if dict["model"] is not None: pbrl.model.device = device_
    print(f"Loaded {fname}")
    return pbrl
