import networkx as nx
from torch import tensor, isnan, zeros


class PreferenceGraph:
    """
    xxx
    """
    def __init__(self, device, episodes=None):
        self.device = device
        self._graph = nx.DiGraph()
        self._graph.add_nodes_from([(i, {"transitions": ep}) for i, ep in
                                   enumerate(episodes if episodes is not None else [])])

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

    def add_episode(self, ep_num, transitions):
        self._graph.add_node(ep_num, transitions=transitions)

    def add_preference(self, history_key, i, j, preference):
        assert (not self._graph.has_edge(i, j)) and (not self._graph.has_edge(j, i)), f"Already have preference for ({i}, {j})"
        self._graph.add_edge(i, j, history_key=history_key, preference=preference)

    def rewind(self, history_key):
        """
        Create an edge_subgraph based on history_key.
        """
        if history_key is None: return self
        g = PreferenceGraph(device=self.device)
        g._graph = self._graph.edge_subgraph((u,v) for u,v,d in self._graph.edges(data=True)
                                             if d["history_key"] <= history_key)
        return g

    def construct_A_and_y(self):
        """
        Construct A and y matrices from the preference graph.
        """
        pairs, y, connected = [], [], set()
        for i, j, data in self._graph.edges(data=True):
            pairs.append([i, j]); y.append(data["preference"]); connected = connected | {i, j}
        y = tensor(y, device=self.device).float()
        connected = sorted(list(connected))
        A = zeros((len(pairs), len(connected)), device=self.device)
        i_list, j_list = [], []
        for l, (i, j) in enumerate(pairs): 
            i_c, j_c = connected.index(i), connected.index(j) # Indices in connected list
            A[l, i_c], A[l, j_c] = 1, -1
            i_list.append(i_c); j_list.append(j_c)
        return A, y, i_list, j_list, connected
