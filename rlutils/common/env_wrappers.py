import gym


class NormaliseActionWrapper(gym.ActionWrapper):
    """
    For environments with continuous action spaces.
    Maps normalised actions in [-1,1] into the range used by the environment.
    """
    def __init__(self, env):
        super().__init__(env)
        self.act_k = (self.action_space.high - self.action_space.low) / 2.
        self.act_k_inv = 2./(self.action_space.high - self.action_space.low)
        self.act_b = (self.action_space.high + self.action_space.low) / 2.

    def action(self, action): 
        return self.act_k * action + self.act_b

    def reverse_action(self, action): return self.act_k_inv * (action - self.act_b)


class CustomRewardWrapper(gym.Wrapper): 
    """
    Enables implementation of a custom reward function.

    Reward function should output:
        - Scalar reward.
        - Boolean done flag.
        - Info dictionary. NOTE: Use "reward_components" key to enable wandb monitoring.
    """
    def __init__(self, env, R=None):
        self.env = env
        super().__init__(self.env)
        if R is not None: self.R = R # This is the reward function.

    def R(_, __, ___, reward, done, ____): return reward, done, {} # Default if None.

    def reset(self): self.state = self.env.reset().copy(); return self.state
        
    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        reward, done, info_add = self.R(self.state, next_state, action, reward, done, info)
        self.state = next_state.copy()
        return self.state, reward, done, {**info, **info_add}


class MetaFeatureWrapper(gym.Wrapper):
    """
    Constructs a dictionary of additional observation features, that are *not* given to the agent, but are instead appended to info. 
    Has access to all historical (state, action, reward, info) tuples from the current episode.
    """
    def __init__(self, env, f):
        self.env = env
        super().__init__(self.env)
        self.f = f # This is the feature constructor function.
    
    def reset(self): 
        state = self.env.reset()
        self.states, self.actions, self.rewards, self.infos = [state], [], [], []
        return state

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        self.states.append(next_state) 
        self.actions.append(action)
        self.rewards.append(reward)
        self.infos.append(info)
        info_add = self.f(self.states, self.actions, self.rewards, self.infos)
        return next_state, reward, done, {**info, **info_add}