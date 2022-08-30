def preference_batch(sampler, interface, graph, batch_size, ij_min=0, history_key=None, budget=float("inf")):
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
                graph.add_preference(i, j, preference, history_key=history_key)
                readout = f"{sampler._k} / {batch_size} ({len(graph.edges)} / {budget}): P({i} > {j}) = {preference}"
                print(readout); interface.print("\n"+readout)
            elif exit_code == 1: print("=== Batch complete ==="); break
            elif exit_code == 2: print("=== Fully connected ==="); break
    return {"feedback_count": len(graph.edges)}

