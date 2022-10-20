class SumLogger:
    def __init__(self, P): 
        self.P = P
        self.run_names = []

    def per_timestep(self, ep, t, state, action, next_state, reward, done, info, extra):
        if   self.P["name"] == "return":    v = reward
        elif self.P["name"] == "ep_length": v = 1
        elif self.P["source"] == "info":    v = info[self.P["key"]]
        elif self.P["source"] == "extra":   v = extra[self.P["key"]]
        if t == 0: self.sum = v
        else: self.sum += v
        
    def per_episode(self, ep): 
        try:    return {f"{self.P['name']}_{i}": v for i, v in enumerate(self.sum)} # If iterable
        except: return {self.P["name"]: self.sum} # Otherwise
