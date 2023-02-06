import wandb
from tqdm import tqdm
from numpy import array, pad, transpose
from pandas import DataFrame


def get(project_name, metrics, filters=None, tag=None):
    if filters is None: filters = [{"": None}]  # Null filter (selects all)
    data = [{"config": [], "metrics": []} for _ in filters]
    # TODO: Can apply filters directly to api.runs
    for run in tqdm(wandb.Api().runs(project_name)):
        if tag is None or tag in run.tags:
            config = run.config
            config.update({"name": run.name, "group": run.group, "job_type": run.job_type})
            active_filters = set()
            for i, f in enumerate(filters):
                activate = True
                for k, v_filter in f.items():
                    v = config; present = True
                    for k_sub in k.split("."):
                        try: v = v[k_sub]
                        except: present = False; break
                    # NOTE: Filter stays active if not present but v_filter == None
                    if (not(present) and v_filter is not None) or (present and v != v_filter):
                        activate = False; break
                if activate: active_filters.add(i)
            if active_filters:
                run_metrics = []
                for step in tqdm(run.scan_history(keys=metrics), leave=False):
                    if any("video." in m for m in step): continue # NOTE: Skip video logging steps
                    run_metrics.append([step[m] if m in step else float("nan") for m in metrics])
                for i in active_filters:
                    data[i]["config"].append(config)
                    data[i]["metrics"].append(run_metrics)    
    dataframes = [{} for _ in filters]
    for i, f in enumerate(filters):
        if len(data[i]["config"]) == 0:
            print(f"Warning: filter {f} returned no runs"); continue
        # Pad with NaNs to ensure that all the same length
        max_length = max(len(r) for r in data[i]["metrics"])
        data_padded = array([pad(r, ((0,max_length-len(r)),(0,0)), constant_values=float("nan"))
                             for r in data[i]["metrics"]])
        for m, data_m in zip(metrics, transpose(data_padded)):
            dataframes[i][m] = {
                "fname": "-".join([f"{k}={v}" for k,v in f.items()]) + "---" + m + ".csv",
                "df": DataFrame(data_m, columns=[c["name"] for c in data[i]["config"]]),
                "config": data[i]["config"],
            }
    return dataframes
