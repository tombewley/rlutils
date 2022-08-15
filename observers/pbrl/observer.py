from ...rewards.graph import PreferenceGraph
from ...rewards.sampler import Sampler
from ...rewards.interactions import preference_batch
from ...rewards.evaluate import preference_loss, epic, rank_correlation
from .explainer import Explainer

import os
from torch import save as pt_save, load as pt_load, device, cuda


class PbrlObserver:
    """
    Class that manages the interactions between a preference graph, sampler, interface, model and agent
    to enable online preference-based reinforcement learning. Intended to be as lightweight as possible.
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
            if "offline_graph_path" in self.P:
                self.offline_graph = pt_load(self.P["offline_graph_path"], map_location=self.graph.device)
                self.offline_graph.device = self.graph.device
            else: self.offline_graph = None
        self._current_ep = []
    
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
        Operations to complete at the end of an episode, which may include adding this episode
        to the preference graph, creating logs, and (if self._online==True), occasionally gathering
        a preference batch and updating the reward model.
        """   
        states, actions, next_states = self.graph.tensorise(self._current_ep)
        self._current_ep = []
        logs = {}
        ep_info = {"run_name": self.run_names[-1], "ep_num": ep_num}
        # Log reward sums
        if self.P["reward_source"] == "model": 
            logs["model_return"] = self.model.fitness(states, actions, next_states)[0].item()
        if self.interface is not None and self.interface.oracle is not None:
            ep_info["oracle_rewards"] = self.interface.oracle(states, actions, next_states)
            ep_info["oracle_return"] = logs["oracle_return"] = sum(ep_info["oracle_rewards"]).item()
        if self._observing and (ep_num+1) % self.P["observe_freq"] == 0:
            # Add episodes to the preference graph with a specified frequency
            # NOTE: Nodes are numbered as consecutive integers, ep_num stored in ep_info
            self.graph.add_episode(states, actions, next_states, **ep_info)
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
                # Gather preference batch
                logs.update(preference_batch(
                    sampler=self.sampler,
                    interface=self.interface,
                    graph=self.graph,
                    batch_size=batch_size,
                    ij_min=self._n_on_prev_batch,
                    history_key=(ep_num+1),
                    budget=self.P["feedback_budget"]
                ))
                if self.graph.edges:
                    # Update reward model
                    logs.update(self.model.update(graph=self.graph, mode="preference", history_key=(ep_num+1)))
                    # Evaluate by preference loss
                    loss_bce, loss_0_1 = preference_loss([self.model], self.graph)
                    logs["online_preference_loss_bce"], logs["online_preference_loss_0-1"] = loss_bce[0].item(), loss_0_1[0].item()
                    if self.offline_graph is not None:
                        loss_bce, loss_0_1 = preference_loss([self.model], self.offline_graph)
                        logs["offline_preference_loss_bce"], logs["offline_preference_loss_0-1"] = loss_bce[0].item(), loss_0_1[0].item()
                    if self.interface.oracle is not None:
                        # Evaluate by return, reward and rank correlation correlation
                        corr_r, corr_g, _, _ = epic([self.interface.oracle, self.model], self.graph)
                        logs["online_reward_correlation"], logs["online_return_correlation"] = corr_r[0,1].item(), corr_g[0,1].item()
                        logs["online_rank_correlation"] = rank_correlation([self.interface.oracle, self.model], self.graph)[0,1]
                        if self.offline_graph is not None:
                            corr_r, corr_g, _, _ = epic([self.interface.oracle, self.model], self.offline_graph)
                            logs["offline_reward_correlation"], logs["offline_return_correlation"] = corr_r[0,1].item(), corr_g[0,1].item()
                            logs["offline_rank_correlation"] = rank_correlation([self.interface.oracle, self.model], self.offline_graph)[0,1]
                    self.relabel_memory() # If applicable, relabel the agent's replay memory using the updated reward
                self._batch_num += 1 
                self._n_on_prev_batch = len(self.graph)
            # Periodically log and save out
            if self.explainer.P and (ep_num+1) % self.explainer.P["freq"] == 0: self.explainer(history_key=(ep_num+1))
        if self._saving and (ep_num+1) % self.P["save_freq"] == 0: self.save(history_key=(ep_num+1))
        return logs

    def relabel_memory(self): pass

    def save(self, history_key, zfill=4):
        path = f"{self.P['save_path']}/{self.run_names[-1]}"
        if not os.path.exists(path): os.makedirs(path)
        pt_save(self.graph, f"{path}/{str(history_key).zfill(zfill)}_{len(self.graph)}e_{len(self.graph.edges)}p.graph")
        pt_save(self.model, f"{path}/{str(history_key).zfill(zfill)}.reward")

def load(graph_fname=None, model_fname=None, P=None):
    """
    Make an instance of PbrlObserver from the information stored by the .save() method.
    """
    device_ = device("cuda" if cuda.is_available() else "cpu")
    pbrl = PbrlObserver(P)
    if graph_fname is not None:
        pbrl.graph = pt_load(graph_fname, device_)
        pbrl.graph.device = device_
    if model_fname is not None:
        pbrl.model = pt_load(model_fname, device_)
        pbrl.model.device = device_
    print(f"Loaded graph {graph_fname} and model {model_fname}")
    return pbrl
