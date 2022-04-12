from torch import cat


def preference_batch(sampler, interface, graph, batch_size, ij_min, history_key, budget=float("inf")):
    """
    Sample a batch of trajectory pairs, collect preferences via an interface, and add them to a graph.
    """
    sampler.batch_size, sampler.ij_min = batch_size, ij_min
    with interface:
        for exit_code, i, j, _ in sampler:
            if exit_code == 0:
                preference = interface(i, j)
                if preference == "esc": print("=== Feedback exited ==="); break
                elif preference == "skip": print(f"({i}, {j}) skipped"); continue
                graph.add_preference(history_key, i, j, preference)
                readout = f"{sampler._k} / {batch_size} ({len(graph.edges)} / {budget}): P({i} > {j}) = {preference}"
                print(readout); interface.print("\n"+readout)
            elif exit_code == 1: print("=== Batch complete ==="); break
            elif exit_code == 2: print("=== Fully connected ==="); break
    return {"feedback_count": len(graph.edges)}

def update_model(graph, featuriser, model, history_key):
    """
    Update a reward model using the provided preference graph and featuriser.
    """
    # Assemble data structures needed for learning
    A, y, i_list, j_list, connected = graph.construct_A_and_y()
    print(f"Connected episodes: {len(connected)} / {len(graph)}")
    if len(connected) == 0: print("=== None connected ==="); return {}
    # Get lengths and apply feature mapping to all episodes that are connected to the preference graph
    connected_ep_transitions = [graph.nodes[i]["transitions"] for i in connected]
    ep_lengths = [len(tr) for tr in connected_ep_transitions]
    features = featuriser(cat(connected_ep_transitions))
    # Update the reward model using connected episodes
    logs = model.update(history_key, features, ep_lengths, A, i_list, j_list, y)
    return logs
