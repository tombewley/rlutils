import networkx as nx
from numpy.random import default_rng
from torch import tensor, isnan, zeros
import matplotlib.pyplot as plt


class PreferenceGraph:
    """
    xxx
    """
    def __init__(self, device, episodes=None):
        self.device = device
        self._graph = nx.DiGraph()
        self._graph.add_nodes_from([(i, {"transitions": ep}) for i, ep in
                                    enumerate(episodes if episodes is not None else [])])
        self.seed()

    def seed(self, seed=None):
        self.rng = default_rng(seed)

    def __len__(self): return len(self._graph)

    @property
    def nodes(self): return self._graph.nodes

    @property
    def edges(self): return self._graph.edges

    @property
    def preference_matrix(self):
        matrix = tensor(nx.to_numpy_matrix(self._graph, weight="preference", nonedge=float("nan")), device=self.device).float()
        reverse = (1 - matrix).T; mask = ~isnan(reverse)
        matrix[mask] = reverse[mask]
        return matrix

    def add_episode(self, transitions, **kwargs):
        """
        NOTE: Currently numbers nodes as consecutive integers, but stores ep_num as an attribute.
        """
        self._graph.add_node(len(self._graph), transitions=transitions, **kwargs)

    def add_preference(self, history_key, i, j, preference):
        assert i in self._graph and j in self._graph, "Invalid episode index"
        assert (not self._graph.has_edge(i, j)) and (not self._graph.has_edge(j, i)), f"Already have preference for ({i}, {j})"
        self._graph.add_edge(i, j, history_key=history_key, preference=preference)

    def make_data_structures(self, unconnected_ok=False):
        """
        Starting with the preference graph, assemble data structures needed for model updates.
        """
        connected_subgraph = self.subgraph(self._graph.edges) # Remove unconnected episodes
        print(f"Connected episodes: {len(connected_subgraph)} / {len(self)}")
        if not(unconnected_ok) and len(connected_subgraph) > 0:
            assert nx.is_weakly_connected(connected_subgraph._graph), "Morrissey-Gulliksen requires a connected graph."
        connected = sorted(connected_subgraph.nodes)
        transitions = [connected_subgraph.nodes[i]["transitions"] for i in connected]
        pairs = list(connected_subgraph.edges)
        y = tensor([connected_subgraph.edges[ij]["preference"] for ij in pairs], device=self.device).float()
        A = zeros((len(pairs), len(connected)), device=self.device)
        i_list, j_list = [], []
        for l, (i, j) in enumerate(pairs): 
            i_c, j_c = connected.index(i), connected.index(j) # Indices in connected list
            A[l, i_c], A[l, j_c] = 1, -1
            i_list.append(i_c); j_list.append(j_c)
        return transitions, A, i_list, j_list, y

    def show(self, figsize=(12, 12)):
        # self.pbrl.graph = nx.DiGraph()
        # n = len(self.episodes)
        # self.pbrl.graph.add_nodes_from(range(n), fitness=np.nan, fitness_cv=np.nan)
        # for i in range(n):
        #     if self.episodes[i] is not None: self.pbrl.graph.nodes[i]["fitness"], _ = self.pbrl.fitness(self.pbrl.episodes[i])
        # for i, f in zip(self._connected, self._ep_fitness_cv):
        #     self.pbrl.graph.nodes[i]["fitness_cv"] = f * len(self.episodes[i])
        # self.pbrl.graph.add_weighted_edges_from([(j, i, self.preferences[i,j]) for i in range(n) for j in range(n) if not np.isnan(self.preferences[i,j])])
        # fitness = list(nx.get_node_attributes(self.pbrl.graph, "fitness").values())
        # fitness_cv = list(nx.get_node_attributes(self.pbrl.graph, "fitness_cv").values())
        # vmin, vmax = min(np.nanmin(fitness), np.nanmin(fitness_cv)), max(np.nanmax(fitness), np.nanmax(fitness_cv))
        # Plot nodes
        plt.figure(figsize=figsize)
        pos = nx.drawing.nx_agraph.graphviz_layout(self._graph, prog="neato")
        nx.draw_networkx_nodes(self._graph, pos=pos,
            node_size=500,
            # node_color=fitness_cv,
            # cmap="coolwarm_r", vmin=vmin, vmax=vmax
        )
        node_collection = nx.draw_networkx_nodes(self._graph, pos=pos,
            node_size=250,
            # node_color=fitness,
            # cmap="coolwarm_r", vmin=vmin, vmax=vmax,
            linewidths=1, edgecolors="w"
        )
        nx.draw_networkx_labels(self._graph, pos=pos)
        # Plot edges
        edge_collection = nx.draw_networkx_edges(self._graph, pos=pos,
            node_size=500,
            width=2, arrowsize=20,
            edge_color=list(nx.get_edge_attributes(self._graph, "preference").values()),
            connectionstyle="arc3,rad=0.1",
            edge_cmap=plt.cm.coolwarm_r
        )
        # weights = list(nx.get_edge_attributes(self._graph, "preference").values())
        # for i, e in enumerate(edge_collection): e.set_alpha(weights[i])
        # nx.draw_networkx_edge_labels(self._graph, pos=pos, label_pos=0.4, font_size=6,
        #     edge_labels={(i, j): f"{d['weight']:.2f}" for i, j, d in self._graph.edges(data=True)}
        #     )
        return node_collection, edge_collection

# ==============================================================================
# SUBGRAPH EXTRACTION

    def subgraph(self, edges):
        sg = PreferenceGraph(device=self.device)
        sg._graph = self._graph.edge_subgraph(edges)
        return sg

    def rewind_subgraph(self, history_key):
        """
        Create a frozen subgraph containing all preferences added before history_key.
        """
        if history_key is None: return self
        return self.subgraph((i,j) for i,j,d in self._graph.edges(data=True) if d["history_key"] <= history_key)

    def random_connected_subgraph(self, num_edges):
        """
        Create a frozen connected subgraph with a specified number of preferences.
        Adapted from pseudocode in https://stackoverflow.com/a/64814482/.
        TODO: Different selection strategies (e.g. prioritise pairs already in subgraph to increase density).
        """
        node = self.rng.choice([node for node in self.nodes if self._graph.degree(node) > 0]) # Random seed node
        edge_queue = set(self._graph.in_edges(node)) | set(self._graph.out_edges(node))
        nodes, edges = set(), set()
        for _ in range(num_edges):
            edge = tuple(self.rng.choice(tuple(edge_queue))) # Random connected edge
            edge_queue.remove(edge); edges.add(edge)
            for node in edge:
                if node not in nodes:
                    edge_queue.update((set(self._graph.in_edges(node)) |
                                       set(self._graph.out_edges(node))) - {edge})
                    nodes.add(node)
        return self.subgraph(edges), self.subgraph([e for e in self.edges if e not in edges])
