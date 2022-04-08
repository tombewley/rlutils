import importlib


# TODO: Wheel reinvention here! Use wandb.config


def recursive_update(d1, d2, i=None, block_overwrite=False, verbose=False):
    # Adapted from https://stackoverflow.com/a/38504949.
    def _update(d1, d2, path=[]):
        for k in d2:
            typ = None
            if k in d1:
                if isinstance(d1[k], dict) and isinstance(d2[k], dict): _update(d1[k], d2[k], path+[k])
                elif block_overwrite: raise Exception(f"{'.'.join(path+[k])}: {d1[k]} | {d2[k]}")
                else: typ = "UP "
            else: typ = "NEW"
            if typ is not None: 
                d1[k] = d2[k]
                if verbose: print(f"{typ} {'.'.join(path+[k])}: {d1[k]}")
    def _select_index(d1):
        for k in d1: 
            if isinstance(d1[k], dict): 
                try: d1[k] = d1[k][i]
                except: _select_index(d1[k])
            elif isinstance(d1[k], list): d1[k] = d1[k][int(i)]
    _update(d1, d2)
    if i is not None: _select_index(d1)


def build_params(paths, params=None, root_dir="", verbose=False):
    if params is None: params = {}
    paths = [p.replace("/",".") for p in paths]
    root_dir = root_dir.replace("/", ".")
    update = {}
    for p in paths:
        i = None
        try:
            if "=" in p: # For parameter array
                p, i = p.split("=")
            new = importlib.import_module(f"{root_dir}.{p}").P
        except ImportError: # If not a recognised config file, treat as filename for loading
            new = {"deployment": {"agent_load_fname": p}} 
        recursive_update(update, new, i=i, block_overwrite=True, verbose=False)
    recursive_update(params, update, verbose=verbose)
    return params
