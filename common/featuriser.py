from torch import cat


class Featuriser:
    """
    Class for mapping multidimensional (s, a, s') tensors to a corresponding tensor of features.
    Passing the dictionary of existing features into each function enables order-dependent construction without repeated calculation.
    """
    def __init__(self, P):
        self.P = P
        self.names = [f.__name__ for f in self.P["features"]] if "features" in self.P else self.P["feature_names"]

    def __call__(self, states, actions, next_states):
        if "preprocessor" in self.P:
            states, actions, next_states = self.P["preprocessor"](states.clone(), actions.clone(), next_states.clone())
        if "features" not in self.P: 
            return states, actions, next_states
        features = {}
        for func, name in zip(self.P["features"], self.names):
            features[name] = func(states, actions, next_states, features)
        return cat([features[name].unsqueeze(-1) for name in self.names], dim=-1)
