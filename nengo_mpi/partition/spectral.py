
_spectral_available = True

try:
    import networkx as nx
    if not hasattr(nx, 'spectral_ordering'):
        raise ImportError()
except ImportError:
    _spectral_available = False


def spectral_available():
    return _spectral_available


def spectral_partitioner(cluster_graph, n_components):
    """
    A heuristic approach to partitioning using a spectral ordering.
    First computes the spectral ordering (which effectively tries to place
    nodes that have many edges between them closer together). Then
    repeatedly chooses the component with the largest number of neurons,
    assigns it to a new component, and assigns all nodes that are nearby int
    the ordering to the same component. We switch to the new component once
    adding another node to the component would give that component too many
    neurons.

    Parameters
    ----------
    cluster_graph: networkx Graph
        A graph created from a network using network_to_cluster_graph.

    n_components: int
        Desired number of components in the partition.

    Returns
    -------
    assignments: dict
        A mapping from nodes in the cluster graph to components.
    """
    assert n_components > 1

    ordering = nx.spectral_ordering(cluster_graph)

    total_neurons = sum(
        n.n_neurons
        for n in cluster_graph.nodes())

    neurons_per_component = float(total_neurons) / n_components

    component = 0
    assignments = {}
    while ordering:
        next_index = max(
            range(len(ordering)),
            key=lambda i: ordering[i].n_neurons)

        component_n_neurons = 0
        while ordering and component_n_neurons < neurons_per_component:
            assignments[ordering[next_index]] = component

            component_n_neurons += ordering[next_index].n_neurons

            del ordering[next_index]

            if next_index >= len(ordering):
                next_index = len(ordering) - 1

        component += 1

    return assignments
