from torch import cat


class Featuriser:
    """
    Class for mapping a multidimensional tensor of transitions to a corresponding tensor of features.
    """
    def __init__(self, P):
        self.P = P
        self.names = [f.__name__ for f in self.P["features"]]

    def __call__(self, transitions):
        if "preprocessor" in self.P: transitions = self.P["preprocessor"](transitions.clone()) # Prevent in-place modifications
        features = {}
        for func, name in zip(self.P["features"], self.names):
            # Passing features dict into each func allows recursive construction without repeated calculation
            features[name] = func(transitions, features)
        return cat([features[name].unsqueeze(-1) for name in self.names], dim=-1)
