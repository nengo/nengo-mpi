from collections import defaultdict
import warnings

import networkx as nx
import numpy as np
from nengo import Direct, Node, Ensemble
from nengo.ensemble import Neurons
from nengo.utils.builder import find_all_io

from .spectral import spectral_partitioner

import logging
logger = logging.getLogger(__name__)


def verify_assignments(network, assignments):
    """
    Propogate the assignments given in ``assignments'' to form a complete
    partition of the network, and verify that the resulting partition is
    usable.

    network: nengo.Network
        The network whose assignments are to be verified.

    assignments: dict
        A mapping from each nengo object in the network to an integer
        specifiying which component of the partition the object is assigned
        to. Component 0 is simulated by the master process.
    """

    propogate_assignments(network, assignments)
    n_components = max(assignments.values()) + 1

    return n_components, assignments


class Partitioner(object):
    """
    A class for dividing a nengo network into components. Connections that
    straddle component boundaries must be filtered connections.

    Parameters
    ----------
    n_components: int
        The number of components to divide the nengo network into. If None,
        defaults to 1.

    func: A function to partition the nengo graph, assigning nengo objects
        to component indices. Ignored if n_components == 1.

        Arguments:
            filter_graph
            n_components

    args: Extra positional args passed to func

    kwargs: Extra keyword args passed to func
    """

    def __init__(
            self, n_components=None, func=None, *args, **kwargs):

        if n_components is None:
            self.n_components = 1

            if func is not None:
                warnings.warn(
                    "Number of components not specified. Defaulting to "
                    "1 and ignoring supplied partitioner function.")

            self.func = self.default_partition_func

        else:
            self.n_components = n_components

            if func is None:
                func = self.default_partition_func

            self.func = func

        self.args = args
        self.kwargs = kwargs

    @property
    def default_partition_func(self):
        return spectral_partitioner

    def partition(self, network):
        """
        Partition the network using the partitioning function self.func.
        If self.n_components == 1 or the number of independently
        simulatable chunks of the network is less than self.n_components, the
        partitioning function is not used. In the former case, all objects
        go on component 0, and in the latter case, we arbitrarily assign each
        independently simulatable chunk to its own component.

        Parameters
        ----------
        network: nengo.Network
            The network to partition.

        Returns
        -------
        n_components: int
            The number of components the nengo network is split into. May
            be different than the value supplied to Partitioner.__init__ in
            cases where it is deemed impossible to split the network into the
            desired number of components.

        assignments: dict
            A mapping from each nengo object in the network to an integer
            specifiying which component of the partition the object is assigned
            to. Component 0 is simulated by the master process.
        """

        object_assignments = {}

        if self.n_components > 1:
            component0, filter_graph = network_to_filter_graph(network)

            n_nodes = len(filter_graph)

            if n_nodes <= self.n_components:
                self.n_components = n_nodes
                node_assignments = {
                    node: i for i, node in enumerate(filter_graph)}
            else:
                node_assignments = self.func(
                    filter_graph, self.n_components,
                    *self.args, **self.kwargs)

            if component0:
                # Assign the node ``component0'' to component 0
                on_zero = filter(
                    lambda n: node_assignments[n] == 0, node_assignments)
                c = node_assignments[component0]
                on_c = filter(
                    lambda n: node_assignments[n] == c, node_assignments)

                node_assignments.update({node: c for node in on_zero})
                node_assignments.update({node: 0 for node in on_c})

            for node in node_assignments:
                node.assign_to_component(
                    object_assignments, node_assignments[node])

            evaluate_partition(
                network, self.n_components, object_assignments, filter_graph)

        propogate_assignments(network, object_assignments)

        return self.n_components, object_assignments


def network_to_filter_graph(network):
    """
    Creates a graph from a nengo network, where the nodes are collections
    of nengo objects that are connected by non-filtered connections, and edges
    are filtered connections between those components. More precisely, two
    nengo objects are in the same node of this higher order graph iff there
    exists a path between them in the undirected graph of nengo objects which
    does not contain a filter. This is required for partitioning a network
    for use by nengo_mpi, since we can only send data across nengo connections
    that contain an update operation, which means they must be filtered.

    Parameters
    ----------
    network: The nengo network to partition.

    Returns
    -------
    component0: GraphNode
        A GraphNode containing all nengo objects which must be simulated on the
        master node in the nengo_mpi simulator. If there are no such objects,
        then this has value None.

    filter_graph: networkx Graph
        Where the nodes are intances of GraphNode. Importantly, the filter
        GraphNode contains the node ``component0'' if ``component0'' is not
        None.
    """

    def merge_nodes(node_map, a, b):
        if a.merge(b):
            for obj in b.objects:
                node_map[obj] = a

            del b

        return a

    node_map = defaultdict(GraphNode)

    for conn in network.all_connections:

        pre_obj = neurons2ensemble(conn.pre_obj)

        pre_node = node_map[pre_obj]

        if pre_node.empty():
            pre_node.add_object(pre_obj)

        post_obj = neurons2ensemble(conn.post_obj)

        post_node = node_map[post_obj]

        if post_node.empty():
            post_node.add_object(post_obj)

        if is_update(conn):
            pre_node.add_output(conn)
            post_node.add_input(conn)
        else:
            merge_nodes(node_map, pre_node, post_node)

    _, outputs = find_all_io(network.all_connections)

    # merge together all nodes that have to go on component 0
    component0 = filter(
        lambda x: for_component0(x, outputs), node_map.values())

    if component0:
        component0 = reduce(
            lambda u, v: merge_nodes(node_map, u, v), component0)
    else:
        component0 = None

    G = nx.Graph()

    update_connections = filter(is_update, network.all_connections)

    for conn in update_connections:
        pre_node = node_map[neurons2ensemble(conn.pre_obj)]
        post_node = node_map[neurons2ensemble(conn.post_obj)]

        if pre_node != post_node:
            if G.has_edge(pre_node, post_node):
                G[pre_node][post_node]['weight'] += conn.size_mid
            else:
                G.add_edge(pre_node, post_node, weight=conn.size_mid)

    return component0, G


class GraphNode(object):
    """
    A class to use for nodes in the filter graph which is created as part
    of the partitioning process. Represents a group of nengo obejects
    which must be simulated on the same processor in nengo_mpi.
    """

    def __init__(self):
        self.objects = set()
        self.inputs = set()
        self.outputs = set()
        self._n_neurons = 0

    @property
    def n_neurons(self):
        return self._n_neurons

    def add_object(self, obj):
        self.objects.add(obj)

        if hasattr(obj, 'n_neurons'):
            self._n_neurons += obj.n_neurons

    def add_input(self, i):
        self.inputs.add(i)

    def add_output(self, o):
        self.outputs.add(o)

    def empty(self):
        return len(self.objects) == 0

    def __str__(self):
        s = ",".join(str(o) for o in self.objects)
        return s

    def assign_to_component(self, assignments, component):
        """
        Assign all nengo objects in this GraphNode to the given component.
        Alters the provided ``assignments'' dictionary. Returns None.

        Parameters
        ----------
        assignments: dict
            A dict mapping each nengo objects to its component.

        component: intent
            The component to assign the nengo objects to.
        """

        for obj in self.objects:
            assignments[obj] = component

    def merge(self, other):
        """Return True if a merging occurs."""

        if self == other:
            return False

        for obj in other.objects:
            if hasattr(obj, 'n_neurons'):
                self._n_neurons += obj.n_neurons

        self.objects = self.objects.union(other.objects)
        self.inputs = self.inputs.union(other.inputs)
        self.outputs = self.outputs.union(other.outputs)

        return True


def is_update(conn):
    return conn.synapse is not None


def neurons2ensemble(e):
    return e.ensemble if isinstance(e, Neurons) else e


def for_component0(node, outputs):
    """Returns whether the component must be simulated on process 0."""

    for obj in node.objects:
        if isinstance(obj, Node) and callable(obj.output):
            return True

        if isinstance(obj, Node):
            if any([conn.function is not None for conn in outputs[obj]]):
                return True

        if isinstance(obj, Ensemble) and isinstance(obj.neuron_type, Direct):
            return True

    return False


def propogate_assignments(network, assignments):
    """
    Propogates the component assignments stored in the dict ``assignments''
    (which only needs to contain assignments for top level networks, nodes and
    ensembles) down to objects that are contained in those top-level objects.
    If assignments is empty, then all objects will be assigned to the 1st
    component, which has index 0. The intent is to have some partitioning
    algorithm determine some of the assignments before this function is called,
    and then this function propogates those assignments.

    Also does a small amount of validation, making sure that certain types of
    objects are assigned to the master component (component 0), and making sure
    that connections that straddle component boundaries have a filter on them.

    These objects are:
        1. Nodes with callable outputs.
        2. Ensembles of Direct neurons.
        3. Any node that is the source of a Connection that has a function.

    Parameters
    ----------
    network: nengo.Network
        The network we are partitioning.

    assignments: dict
        A dictionary mapping from nengo objects to component indices.
        This dictionary will be altered to contain assignments for all objects
        in the network. If a network appears in assignments, then all objects
        in that network which do not also appear in assignments will be given
        the same assignment as the network.

    Returns
    -------
    Nothing, but ``assignments'' is modified.

    """
    def helper(network, assignments, outputs):
        """
        outputs: a dict mapping each nengo object to its output connections.
        """
        for node in network.nodes:
            if callable(node.output):
                if node in assignments and assignments[node] != 0:
                    warnings.warn(
                        "Found Node with callable output was assigned to a "
                        "component other than component 0. Overriding "
                        "previous assignment.")

                assignments[node] = 0

            else:
                if any([conn.function is not None for conn in outputs[node]]):
                    import pdb
                    pdb.set_trace()

                    if node in assignments and assignments[node] != 0:
                        warnings.warn(
                            "Found Node with an output connection whose "
                            "function is not None, which is assigned to a "
                            "component other than component 0. Overriding "
                            "previous assignment.")

                    assignments[node] = 0

                elif node not in assignments:
                    assignments[node] = assignments[network]

        for ensemble in network.ensembles:
            if isinstance(ensemble.neuron_type, Direct):
                if ensemble in assignments and assignments[ensemble] != 0:
                    warnings.warn(
                        "Found Direct-mode ensemble that was assigned to a "
                        "component other than component 0. Overriding "
                        "previous assignment.")

                assignments[ensemble] = 0

            elif ensemble not in assignments:
                assignments[ensemble] = assignments[network]

            assignments[ensemble.neurons] = assignments[ensemble]

        for n in network.networks:
            if n not in assignments:
                assignments[n] = assignments[network]

            helper(n, assignments, outputs)

    def probe_helper(network, assignments):
        # TODO: properly handle probes that target connections
        # connections will not be in ``assignments'' at this point.
        for probe in network.probes:
            assignments[probe] = assignments[probe.target]

        for n in network.networks:
            probe_helper(n, assignments)

    assignments[network] = 0

    try:
        _, outputs = find_all_io(network.all_connections)

        helper(network, assignments, outputs)
        probe_helper(network, assignments)
    except KeyError as e:
        # Nengo tests require a value error to be raised in these cases.
        msg = "Invalid Partition. KeyError: %s" % e.message,
        raise ValueError(msg)

    non_updates = [
        conn for conn in network.all_connections if not is_update(conn)]

    for conn in non_updates:
        pre_component = assignments[conn.pre_obj]
        post_component = assignments[conn.post_obj]

        if pre_component != post_component:
            raise RuntimeError(
                "Non-filtered connection %s straddles component "
                "boundaries. Pre-object assigned to %d, post-object "
                "assigned to %d." % (conn, pre_component, post_component))

    nodes = network.all_nodes
    nodes_in = all([node in assignments for node in nodes])
    assert nodes_in, "Assignments incomplete, missing nodes."

    ensembles = network.all_ensembles
    ensembles_in = all([ensemble in assignments for ensemble in ensembles])
    assert ensembles_in, "Assignments incomplete, missing ensembles."

    probes = network.all_probes
    probes_in = all([probe in assignments for probe in probes])
    assert probes_in, "Assignments incomplete, missing probes."


def total_neurons(network):
    n_neurons = 0

    for e in network.ensembles:
        n_neurons += e.n_neurons

    for n in network.networks:
        n_neurons += total_neurons(n)

    return n_neurons


def evaluate_partition(
        network, n_components, assignments, filter_graph):
    """Prints a summary of the quality of a partition."""

    print "*" * 80

    key = lambda n: sum(
        e.n_neurons for e in n.objects if hasattr(e, 'n_neurons'))

    graph_node_n_neurons = [key(n) for n in filter_graph.nodes()]
    graph_node_n_items = [len(n.objects) for n in filter_graph.nodes()]

    print "Filter graph statistics:"
    print "Number of nodes: ", filter_graph.number_of_nodes()
    print "Number of edges: ", filter_graph.number_of_edges()

    print "Mean neurons per filter graph node: ", np.mean(graph_node_n_neurons)
    print "Std of neurons per filter graph node", np.std(graph_node_n_neurons)
    print "Min number of neurons", np.min(graph_node_n_neurons)
    print "Max number of neurons", np.max(graph_node_n_neurons)

    print "Mean items per filter graph node: ", np.mean(graph_node_n_items)
    print "Std of items per filter graph node", np.std(graph_node_n_items)
    print "Min number of items", np.min(graph_node_n_items)
    print "Max number of items", np.max(graph_node_n_items)

    component_neuron_counts = [0] * n_components
    component_item_counts = [0] * n_components

    for ens in network.all_ensembles:
        if ens in assignments:
            component_neuron_counts[assignments[ens]] += ens.n_neurons
            component_item_counts[assignments[ens]] += 1

    for node in network.all_nodes:
        if node in assignments:
            component_item_counts[assignments[ens]] += 1

    mean_neuron_count = np.mean(component_neuron_counts)
    neuron_count_std = np.std(component_neuron_counts)

    print "*" * 20
    print "Evaluating partition of network"

    print "Total number of neurons: ", total_neurons(network)
    print "Mean neurons per cluster: ", mean_neuron_count
    print "Standard deviation of neurons per cluster", neuron_count_std
    print "Min number of neurons", np.min(component_neuron_counts)
    print "Max number of neurons", np.max(component_neuron_counts)
    print (
        "Number of empty partitions",
        n_components - np.count_nonzero(component_neuron_counts))

    mean_item_count = np.mean(component_item_counts)
    item_count_std = np.std(component_item_counts)

    print "*" * 10

    print (
        "Total number of nengo items (nodes and ensembles): ",
        len(network.all_nodes + network.all_ensembles))
    print "Mean items per cluster: ", mean_item_count
    print "Standard deviation of items per cluster", item_count_std
    print "Min number of items", np.min(component_item_counts)
    print "Max number of items", np.max(component_item_counts)

    communication_weight = 0
    total_weight = 0

    for conn in network.all_connections:
        if is_update(conn):
            pre_obj = neurons2ensemble(conn.pre_obj)
            post_obj = neurons2ensemble(conn.post_obj)

            if assignments[pre_obj] != assignments[post_obj]:
                communication_weight += conn.size_mid

            total_weight += conn.size_mid

    print "*" * 10
    print "Number of dimensions that are communicated: ", communication_weight
    print "Total number of filtered dimensions: ", total_weight
    print (
        "Percentage of filtered dimensions that are "
        "communicated: ", float(communication_weight) / total_weight)
