import numpy as np


class SumLogger:
    def __init__(self, P): 
        self.P = P
        self.run_names = []

    def per_timestep(self, ep, t, state, action, next_state, reward, done, info, extra):
        if self.P["source"] == "info": c = info[self.P["key"]]
        elif self.P["source"] == "extra": c = extra[self.P["key"]]
        if t == 0: self.sums = np.array(c)
        else: self.sums += c
        
    def per_episode(self, ep): 
        return {f"{self.P['name']}_{c}": r for c, r in enumerate(self.sums)}