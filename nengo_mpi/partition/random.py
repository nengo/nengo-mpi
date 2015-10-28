import numpy as np

def random_partitioner(cluster_graph, n_components, seed=None):
    """
    Assigns each cluster_graph node to a random component.

    Parameters
    ----------
    cluster_graph: networkx Graph
        A graph created from a network using network_to_cluster_graph.

    n_components: int
        Desired number of components in the partition.

    seed: int
        Seed for the random number generator.

    Returns
    -------
    assignments: dict
        A mapping from nodes in the filter graph to components.
    """
    assert n_components > 1

    rng = np.random.RandomState(seed)
    random_ints = rng.random_integers(0, n_components-1, size=len(cluster_graph))
    assignments = {n: c for n, c in zip(cluster_graph, random_ints)}

    return assignments
