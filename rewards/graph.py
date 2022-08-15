import networkx as nx
from numpy import array
from numpy.random import default_rng
from torch import tensor, float32, isnan, zeros, device, cuda
import matplotlib.pyplot as plt


class PreferenceGraph:
    """
    xxx
    """
    def __init__(self):
        self.device = device("cuda" if cuda.is_available() else "cpu")
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
    def ep_lengths(self): return [len(ep["actions"]) for _,ep in self.nodes(data=True)] # NOTE: So robust to future change to store T+1 states!

    @property
    def preference_matrix(self):
        matrix = tensor(nx.to_numpy_matrix(self._graph, weight="preference", nonedge=float("nan")), device=self.device).float()
        reverse = (1 - matrix).T; mask = ~isnan(reverse)
        matrix[mask] = reverse[mask]
        return matrix

    def tensorise(self, s_a_ns_list):
        return tensor(array([s  for s,_,__ in s_a_ns_list]), dtype=float32, device=self.device), \
               tensor(array([a  for _,a,__ in s_a_ns_list]), dtype=float32, device=self.device), \
               tensor(array([ns for _,_,ns in s_a_ns_list]), dtype=float32, device=self.device)

    def add_episode(self, states, actions, next_states, **kwargs):
        # TODO: Storing both states and next_states is wasteful! Just store T+1 states alongside T actions
        self._graph.add_node(len(self._graph), states=states, actions=actions, next_states=next_states, **kwargs)

    def add_preference(self, i, j, preference, history_key=None):
        assert i != j, f"Self-loop preference: {i} = {j}"
        assert i in self._graph and j in self._graph, f"Invalid episode index: {i}, {j}"
        assert (not self._graph.has_edge(i, j)) and (not self._graph.has_edge(j, i)), f"Already have preference for ({i}, {j})"
        assert type(preference) == float and 0. <= preference <= 1., f"Invalid preference value: {preference}"
        self._graph.add_edge(i, j, history_key=history_key, preference=preference)

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

    def show(self, figsize=(12, 12), model=None):
        vmin, vmax = float("inf"), -float("inf")
        if model is not None:
            model_return = [self.model(s, a, ns).sum() for s, a, ns in zip(self.graph.states, self.graph.actions, self.graph.next_states)]
            vmin, vmax = min(vmin, min(model_return)), max(vmax, max(model_return))
        else: model_return = None
        try:
            oracle_return = list(nx.get_node_attributes(self, "oracle_return").values())
            vmin, vmax = min(vmin, min(oracle_return)), max(vmax, max(oracle_return))
        except: oracle_return = None

        """Outer colour by model return"""
        plt.figure(figsize=figsize)
        pos = nx.drawing.nx_agraph.graphviz_layout(self._graph, prog="neato")
        nx.draw_networkx_nodes(self._graph, pos=pos,
            node_size=500,
            node_color=model_return if model_return is not None else "gray",
            cmap="coolwarm_r", vmin=vmin, vmax=vmax
        )
        """Inner colour by oracle return"""
        node_collection = nx.draw_networkx_nodes(self._graph, pos=pos,
            node_size=250,
            node_color=oracle_return if oracle_return is not None else "gray",
            cmap="coolwarm_r", vmin=vmin, vmax=vmax,
            linewidths=1, edgecolors="w"
        )
        nx.draw_networkx_labels(self._graph, pos=pos)
        """Edge colour by preference value"""
        edge_collection = nx.draw_networkx_edges(self._graph, pos=pos,
            node_size=500,
            width=2, arrowsize=20,
            edge_color=list(nx.get_edge_attributes(self._graph, "preference").values()),
            connectionstyle="arc3,rad=0.1",
            edge_cmap=plt.cm.coolwarm_r, edge_vmin=0., edge_vmax=1.
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
