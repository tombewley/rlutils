from torch import device, load as torch_load
from torch.cuda import is_available

from ..common.utils import get_device
from ..agents._default_hyperparameters import default_hyperparameters


def make(agent, env, hyperparameters=dict()):
    """
    Make an instance of an agent class to train/deploy in env, overwriting default hyperparameters with those provided.
    """
    agent = agent.lower()
    # Special treatment for TD3 (a variant of DDPG).
    if agent == "td3": hyperparameters["td3"] = True; agent = "ddpg"
    assert agent in default_hyperparameters, "Agent type not recognised."
    # Overwrite default hyperparameters where applicable.
    P = default_hyperparameters[agent]
    for k, v in hyperparameters.items(): P[k] = v
    # Load agent class.
    if   agent == "actor_critic":       from ..agents.actor_critic import ActorCriticAgent as agent_class
    elif agent == "ddpg":               from ..agents.ddpg import DdpgAgent as agent_class
    elif agent == "diayn":              from ..agents.diayn import DiaynAgent as agent_class
    elif agent == "dqn":                from ..agents.dqn import DqnAgent as agent_class
    elif agent == "forward_backward":   from ..agents.forward_backward import ForwardBackwardAgent as agent_class
    elif agent == "mbpo":               from ..agents.mbpo import MbpoAgent as agent_class
    elif agent == "off_policy_mc":      from ..agents.off_policy_mc import OffPolicyMCAgent as agent_class
    elif agent == "ppo":                from ..agents.ppo import PpoAgent as agent_class
    elif agent == "random":             from ..agents.random_agent import RandomAgent as agent_class
    elif agent == "reinforce":          from ..agents.reinforce import ReinforceAgent as agent_class
    elif agent == "sac":                from ..agents.sac import SacAgent as agent_class
    elif agent == "pets":               from ..agents.pets import PetsAgent as agent_class
    elif agent == "stable_baselines":   from ..agents.stable_baselines import StableBaselinesAgent as agent_class
    elif agent == "steve":              from ..agents.steve import SteveAgent as agent_class
    elif agent == "treeqn":             from ..agents.treeqn import TreeqnAgent as agent_class
    return agent_class(env, P)

def load(path, env): 
    _device = get_device()
    agent = torch_load(path, map_location=_device)
    agent.device = _device
    agent.env = env
    return agent