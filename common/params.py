import importlib
from copy import deepcopy


# TODO: Wheel reinvention here! Use wandb.config


def recursive_update(d1, d2, i=None, block_overwrite=False, verbose=False):
    def _select_index(d):
        for k in d:
            if isinstance(d[k], dict):
                try: d[k] = d[k][i]
                except: _select_index(d[k])
            elif isinstance(d[k], list): d[k] = d[k][int(i)]
    def _update(d1, d2, path=[]):
        # Adapted from https://stackoverflow.com/a/38504949.
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
    if i is not None: _select_index(d2)
    _update(d1, d2)

def build_params(paths, params=None, root_dir="", verbose=False):
    if params is None: params = {}
    if root_dir: root_dir += "."
    update = {}
    for p in paths:
        if not p: continue # Ignore empty or NoneType
        i = None
        if "=" in p: # For parameter array
            p, i = p.split("=")
        new = deepcopy(importlib.import_module(f"{root_dir}{p}".replace("/",".")).P)
        recursive_update(update, new, i=i, block_overwrite=True, verbose=False)
    recursive_update(params, update, verbose=verbose)
    return params
