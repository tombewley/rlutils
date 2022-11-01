import networkx as nx
from numpy import array
from numpy.random import default_rng
from torch import tensor, float32, isnan, zeros, split, stack, hstack, vstack, no_grad, device, cuda
import matplotlib.pyplot as plt

from ..common.utils import get_device


class PreferenceGraph:
    """
    Class for storing a preference dataset.
    """
    def __init__(self):
        self.device = get_device()
        self._graph = nx.DiGraph()
        self.seed()

    def seed(self, seed=None):
        self.rng = default_rng(seed)

    def __len__(self): return len(self._graph)
    def __repr__(self): return f"Preference graph with {len(self)} episodes and {len(self.edges)} preferences"

    @property
    def nodes(self): return self._graph.nodes
    @property
    def edges(self): return self._graph.edges
    @property
    def states(self): return [ep["states"] for _,ep in self.nodes(data=True)]
    @property
    def actions(self): return [ep["actions"] for _,ep in self.nodes(data=True)]
    @property
    def next_states(self): return [ep["next_states"] for _,ep in self.nodes(data=True)]
    @property
    def oracle_rewards(self): return [ep["oracle_rewards"] for _,ep in self.nodes(data=True)]
    @property
    def oracle_returns(self): return [ep["oracle_return"] for _,ep in self.nodes(data=True)]
    @property
    def ep_lengths(self): return [len(ep["actions"]) for _,ep in self.nodes(data=True)] # NOTE: So robust to future change to store T+1 states!

    @property
    def preference_matrix(self):
        matrix = tensor(nx.to_numpy_matrix(self._graph, weight="preference", nonedge=float("nan")), device=self.device).float()
        reverse = (1 - matrix).T; mask = ~isnan(reverse)
        matrix[mask] = reverse[mask]
        return matrix

    def rewards_by_ep_and_returns(self, reward_functions=["oracle"]):
        with no_grad():
            rewards_by_ep = split(stack([hstack(self.oracle_rewards) if r == "oracle" else
                            r(vstack(self.states), vstack(self.actions), vstack(self.next_states))
                            for r in reward_functions]), self.ep_lengths, dim=1)
            return rewards_by_ep, stack([r.sum(dim=1) for r in rewards_by_ep], dim=1)

    def tensorise(self, s_a_ns_list):
        return tensor(array([s  for s,_,__ in s_a_ns_list]), dtype=float32, device=self.device), \
               tensor(array([a  for _,a,__ in s_a_ns_list]), dtype=float32, device=self.device), \
               tensor(array([ns for _,_,ns in s_a_ns_list]), dtype=float32, device=self.device)

    def add_episode(self, states, actions, next_states, **kwargs):
        # TODO: Storing both states and next_states is wasteful! Just store T+1 states alongside T actions
        self._graph.add_node(len(self._graph), states=states, actions=actions, next_states=next_states, **kwargs)

    def add_preference(self, i, j, preference, info):
        assert i != j, f"Self-loop preference: {i} = {j}"
        assert i in self._graph and j in self._graph, f"Invalid episode index: {i}, {j}"
        assert (not self._graph.has_edge(i, j)) and (not self._graph.has_edge(j, i)), f"Already have preference for ({i}, {j})"
        assert type(preference) == float and 0. <= preference <= 1., f"Invalid preference value: {preference}"
        if "history_key" not in info: info["history_key"] = None
        self._graph.add_edge(i, j, preference=preference, **info)

    def preference_data_structures(self, unconnected_ok=False):
        """
        Assemble data structures needed for preference-based model updates.
        """
        sg = self.subgraph(edges=self.edges) # Remove unconnected episodes
        if not(unconnected_ok) and len(sg) > 0:
            assert nx.is_weakly_connected(sg._graph), "Morrissey-Gulliksen requires a connected graph."
        y = tensor([edge["preference"] for _,_,edge in sg.edges(data=True)], device=self.device).float()
        A = zeros((len(sg.edges), len(sg)), device=self.device)
        sg_nodes = list(sg.nodes)
        i_list, j_list = [], []
        for l, (i, j) in enumerate(sg.edges): 
            i_c, j_c = sg_nodes.index(i), sg_nodes.index(j) # Indices in subgraph
            A[l, i_c], A[l, j_c] = 1, -1
            i_list.append(i_c); j_list.append(j_c)
        return sg.states, sg.actions, sg.next_states, A, i_list, j_list, y

    def show(self, figsize=(12, 12), model_return=None, model=None, do_oracle_return=False,
             cmap_lims=None, label_font_family="sans-serif"):
        vmin, vmax = float("inf"), -float("inf")
        if model_return is None and model is not None:
            model_return = [self.model(s, a, ns).sum() for s, a, ns in zip(self.graph.states, self.graph.actions, self.graph.next_states)]
        if model_return is not None:
            vmin, vmax = min(vmin, min(model_return)), max(vmax, max(model_return))
        if do_oracle_return:
            oracle_return = list(nx.get_node_attributes(self, "oracle_return").values())
            vmin, vmax = min(vmin, min(oracle_return)), max(vmax, max(oracle_return))
        if cmap_lims is None: cmap_lims = (vmin, vmax)

        """Outer colour by model return"""
        plt.figure(figsize=figsize)
        pos = nx.drawing.nx_agraph.graphviz_layout(self._graph, prog="neato")
        node_collection = nx.draw_networkx_nodes(self._graph, pos=pos,
            node_size=500,
            node_color=model_return if model_return is not None else "gray",
            cmap="coolwarm_r", vmin=cmap_lims[0], vmax=cmap_lims[1]
        )
        """Inner colour by oracle return"""
        if do_oracle_return:
            nx.draw_networkx_nodes(self._graph, pos=pos,
                node_size=250,
                node_color=oracle_return,
                cmap="coolwarm_r", vmin=cmap_lims[0], vmax=cmap_lims[1],
                linewidths=1, edgecolors="w"
            )
        nx.draw_networkx_labels(self._graph, pos=pos,
            labels={i: i+1 for i in self.nodes},
            font_family=label_font_family,
            font_size=20
        )
        if False:
            """Edge colour by preference value"""
            edge_collection = nx.draw_networkx_edges(self._graph, pos=pos,
                node_size=500,
                width=2, arrowsize=20,
                edge_color=list(nx.get_edge_attributes(self._graph, "preference").values()),
                # connectionstyle="arc3,rad=0.1",
                edge_cmap=plt.cm.coolwarm_r, edge_vmin=0., edge_vmax=1.
            )
        if True:
            """Edge direction by preference value (requires 0-1 preferences)"""
            edgelist = []
            for i, j, d in self.edges(data=True):
                if d["preference"] == 0: edgelist.append((i, j))
                elif d["preference"] == 1: edgelist.append((j, i))
                else: raise Exception("Only works with 0-1 preferences")
            edge_collection = nx.draw_networkx_edges(self._graph, pos=pos,
                edgelist=edgelist,
                node_size=500,
                width=1.5, arrowsize=15,
                edge_color="#b3b3b3"
            )

        # weights = list(nx.get_edge_attributes(self._graph, "preference").values())
        # for i, e in enumerate(edge_collection): e.set_alpha(weights[i])
        # nx.draw_networkx_edge_labels(self._graph, pos=pos, label_pos=0.4, font_size=6,
        #     edge_labels={(i, j): f"{d['weight']:.2f}" for i, j, d in self.edges(data=True)}
        #     )
        return node_collection, edge_collection

# ==============================================================================
# SUBGRAPH EXTRACTION

    def subgraph(self, nodes=None, edges=None):
        sg = PreferenceGraph()
        sg._graph = self._graph.subgraph(nodes) if nodes is not None else self._graph.edge_subgraph(edges)
        return sg

    def rewind_subgraph(self, history_key):
        """
        Create a frozen subgraph containing all preferences added before history_key.
        """
        if history_key is None: return self
        return self.subgraph(edges=[(i,j) for i,j,d in self.edges(data=True) if d["history_key"] <= history_key])

    def random_connected_subgraph(self, num_edges):
        """
        Create a frozen connected subgraph with a specified number of preferences.
        Adapted from pseudocode in https://stackoverflow.com/a/64814482/.
        TODO: Different selection strategies (e.g. prioritise pairs already in subgraph to increase density).
        """
        node = self.rng.choice([node for node in self.nodes if self._graph.degree(node) > 0]) # Random seed node
        edge_queue = set(self._graph.in_edges(node)) | set(self._graph.out_edges(node))
        nodes, edges = {node}, set()
        for _ in range(num_edges):
            edge = tuple(self.rng.choice(tuple(edge_queue))) # Random connected edge
            edge_queue.remove(edge); edges.add(edge)
            for node in edge:
                if node not in nodes:
                    nodes.add(node)
                    edge_queue.update((set(self._graph.in_edges(node)) |
                                       set(self._graph.out_edges(node))) - {edge})
        return self.subgraph(edges=edges), self.subgraph(edges=[e for e in self.edges if e not in edges])

    def random_nodewise_connected_subgraph(self, num_nodes, partitioned):
        """
        Create a frozen connected subgraph with a specified number of episodes.
        Adapted from pseudocode in https://stackoverflow.com/a/64814482/.
        TODO: Different selection strategies (e.g. prioritise pairs already in subgraph to increase density).
        """
        node = self.rng.choice([node for node in self.nodes if self._graph.degree(node) > 0]) # Random seed node
        edge_queue = set(self._graph.in_edges(node)) | set(self._graph.out_edges(node))
        nodes = {node}
        while len(nodes) < num_nodes:
            edge = tuple(self.rng.choice(tuple(edge_queue))) # Random connected edge
            edge_queue.remove(edge)
            for node in edge:
                if node not in nodes:
                    nodes.add(node)
                    edge_queue.update((set(self._graph.in_edges(node)) |
                                       set(self._graph.out_edges(node))) - {edge})
        sg_a = self.subgraph(nodes=nodes)
        if partitioned:
            sg_b = self.subgraph(nodes=[n for n in self.nodes if n not in sg_a.nodes])
            sg_c = self.subgraph(edges=(set(self.edges) - (set(sg_a.edges) | set(sg_b.edges))))
        else: raise NotImplementedError; sg_b = self.subgraph(edges=[e for e in self.edges if e not in sg_a.edges])
        return sg_a, sg_b, sg_c
