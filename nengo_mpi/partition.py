from collections import defaultdict
import warnings
import os
import subprocess

import networkx as nx
import numpy as np
from nengo import Direct, Node, Ensemble
from nengo.ensemble import Neurons
from nengo.utils.builder import find_all_io

from heapq import heapify, heappush, heappop

import logging
logger = logging.getLogger(__name__)


def top_level_partitioner(network, num_components, assignments=None):
    """
    A partitioner that puts top level subnetworks on different partitions if it
    can. Everything inside a top level subnetwork (except nodes) will go on
    the same partition.  Reasonable if your network if split up into
    subnetworks of relatively equal size.

    Ensembles at top level will be handled in the same way. All Nodes in the
    network will be put on component 0 (even those inside subnetworks).

    Parameters
    ----------
    network: The nengo network to partition.

    num_components: The number of components to divide the nengo graph into.
        If None, appropriate value chosen automatically.

    assignments: A dictionary mapping from nengo objects to component indices,
        with 0 as the first component. Used to hard-assign nengo objects
        to specific components.
    """

    if assignments is not None:
        assignments = assignments.copy()
    else:
        assignments = {}

    if num_components == 1:
        return {}

    component = 0
    for ensemble in network.ensembles:
        if ensemble not in assignments:
            assignments[ensemble] = component
            component = (component + 1) % num_components

    for n in network.networks:
        if network not in assignments:
            assignments[n] = component
            component = (component + 1) % num_components

    return assignments


class PriorityDict(dict):
    """
    Retrieved from: http://code.activestate.com/recipes/
        522995-priority-dict-a-priority-queue-with-updatable-prio/

    Dictionary that can be used as a priority queue.

    Keys of the dictionary are items to be put into the queue, and values
    are their respective priorities. All dictionary methods work as expected.
    The advantage over a standard heapq-based priority queue is
    that priorities of items can be efficiently updated (amortized O(1))
    using code as 'thedict[item] = new_priority.'

    The 'smallest' method can be used to return the object with lowest
    priority, and 'pop_smallest' also removes it.

    The 'sorted_iter' method provides a destructive sorted iterator.
    """

    def __init__(self, *args, **kwargs):
        super(PriorityDict, self).__init__(*args, **kwargs)
        self._rebuild_heap()

    def _rebuild_heap(self):
        self._heap = [(v, k) for k, v in self.iteritems()]
        heapify(self._heap)

    def smallest(self):
        """Return the item with the lowest priority.

        Raises IndexError if the object is empty.
        """

        heap = self._heap
        v, k = heap[0]
        while k not in self or self[k] != v:
            heappop(heap)
            v, k = heap[0]
        return k

    def pop_smallest(self):
        """Return the item with the lowest priority and remove it.

        Raises IndexError if the object is empty.
        """

        heap = self._heap
        v, k = heappop(heap)
        while k not in self or self[k] != v:
            v, k = heappop(heap)
        del self[k]
        return k

    def __setitem__(self, key, val):
        # We are not going to remove the previous value from the heap,
        # since this would have a cost O(n).

        super(PriorityDict, self).__setitem__(key, val)

        if len(self._heap) < 2 * len(self):
            heappush(self._heap, (val, key))
        else:
            # When the heap grows larger than 2 * len(self), we rebuild it
            # from scratch to avoid wasting too much memory.
            self._rebuild_heap()

    def setdefault(self, key, val):
        if key not in self:
            self[key] = val
            return val
        return self[key]

    def update(self, *args, **kwargs):
        # Reimplementing dict.update is tricky -- see e.g.
        # http://mail.python.org/pipermail/python-ideas/2007-May/000744.html
        # We just rebuild the heap from scratch after passing to super.
        super(PriorityDict, self).update(*args, **kwargs)
        self._rebuild_heap()

    def sorted_iter(self):
        """Sorted iterator of the priority dictionary items.

        Beware: this will destroy elements as they are returned.
        """
        while self:
            yield self.pop_smallest()


def greedy_k(S, k, key=None):
    if key is None:
        key = lambda x: x

    components = [set() for i in range(k)]
    sizes = [0] * k

    pq = PriorityDict()

    pq.update({i: 0 for i in range(k)})

    for n in sorted(S, reverse=True, key=key):

        smallest = pq.pop_smallest()

        components[smallest].add(n)
        sizes[smallest] += key(n)
        pq[smallest] = sizes[smallest]

    return components, sizes


def work_balanced_partitioner(network, num_components, assignments=None):
    """
    Tries to give each processor an equal number of neurons to simulate,
    making no attempt to minimize communication between processors.

    Parameters
    ----------
    network: The nengo network to partition.

    num_components: The number of components to divide the nengo graph into.
        If None, appropriate value chosen automatically.

    assignments: A dictionary mapping from nengo objects to component indices,
        with 0 as the first component. Used to hard-assign nengo objects
        to specific components.
    """

    if assignments is not None:
        assignments = assignments.copy()
    else:
        assignments = {}

    if num_components == 1:
        return {}

    component_0, G = network_to_filter_graph(network)

    components, sizes = greedy_k(
        G.nodes(), num_components,
        key=lambda n: sum(
            e.n_neurons for e in n.objects if hasattr(e, 'n_neurons')))

    for i, c in enumerate(components):
        for n in c:
            for obj in n.objects:
                assignments[obj] = i

    return assignments


class GraphNode(object):
    def __init__(self):
        self.objects = set()
        self.inputs = set()
        self.outputs = set()
        self.n_neurons = 0

    def add_object(self, obj):
        self.objects.add(obj)

        if hasattr(obj, 'n_neurons'):
            self.n_neurons += obj.n_neurons

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
        for obj in self.objects:
            assignments[obj] = component

    def merge(self, other):
        """Return whether a merging occurs."""

        if self == other:
            return False

        for obj in other.objects:
            if hasattr(obj, 'n_neurons'):
                self.n_neurons += obj.n_neurons

        self.objects = self.objects.union(other.objects)
        self.inputs = self.inputs.union(other.inputs)
        self.outputs = self.outputs.union(other.outputs)

        return True


def is_update(conn):
    return conn.synapse is not None


def neurons2ensemble(e):
    return e.ensemble if isinstance(e, Neurons) else e


def for_component_0(node, outputs):
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


def network_to_filter_graph(network):
    """
    Creates a graph from a nengo network, where the nodes are collections
    of nengo objects that are connected by non-filtered connections, and edges
    are filtered connections between those components. This is useful for
    partitioning, since we can only send data across connections that contain
    an update operation, which, for now, means they must be filtered.

    Parameters
    ----------
    network: The nengo network to partition.

    Returns
    -------
    A 2-tuple, where the first item is a GraphNode containing all objects which
    have to be simulated on component 0 (None if there are no such objects),
    and the second item is a filter graph in the form of a networkx graph.
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
    component_0 = filter(
        lambda x: for_component_0(x, outputs), node_map.values())

    if component_0:
        component_0 = reduce(
            lambda u, v: merge_nodes(node_map, u, v), component_0)
    else:
        component_0 = None

    G = nx.Graph()

    update_connections = filter(
        is_update, network.all_connections)

    for conn in update_connections:
        pre_node = node_map[neurons2ensemble(conn.pre_obj)]
        post_node = node_map[neurons2ensemble(conn.post_obj)]

        if pre_node != post_node:
            if G.has_edge(pre_node, post_node):
                G[pre_node][post_node]['weight'] += conn.size_mid
            else:
                G.add_edge(pre_node, post_node, weight=conn.size_mid)

    return component_0, G


def stoer_wagner_partitioner(network, num_components, assignments=None):
    """

    Parameters
    ----------
    network: The nengo network to partition.

    num_components: The number of components to divide the nengo graph into.
        If None, appropriate value chosen automatically.

    assignments: A dictionary mapping from nengo objects to component indices,
        with 0 as the first component. Used to hard-assign nengo objects
        to specific components.
    """

    if assignments is not None:
        assignments = assignments.copy()
    else:
        assignments = {}

    if num_components == 1:
        return {}

    component_0, G = network_to_filter_graph(network)

    sizes = {'': len(G)}
    components = {'': G}
    names = ['']

    while len(components) < num_components:
        max_name = max(names, key=sizes.__getitem__)

        G = components[max_name]

        # requires networkx 1.9
        cut_value, partition = nx.stoer_wagner(G)

        names.remove(max_name)

        del sizes[max_name]
        del components[max_name]

        A_name = max_name + "A"
        B_name = max_name + "B"

        names.extend([A_name, B_name])

        components[A_name] = G.subgraph(partition[0])
        sizes[A_name] = len(partition[0])

        components[B_name] = G.subgraph(partition[1])
        sizes[B_name] = len(partition[1])

    for i, g in enumerate(components.values()):
        for node in g.nodes():
            for obj in node.objects:
                assignments[obj] = i

    return assignments


def count_neurons(network):
    def helper(network, counts):
        n_neurons = 0

        for e in network.ensembles:
            n_neurons += e.n_neurons

        counts[network, 0] = n_neurons

        for n in network.networks:
            n_neurons += helper(n, counts)

        counts[network] = n_neurons

        return n_neurons

    counts = {}
    helper(network, counts)

    return counts


def spectral_partitioner(network, num_components, assignments=None):
    """

    Parameters
    ----------
    network: The nengo network to partition.

    num_components: The number of components to divide the nengo graph into.
        If None, appropriate value chosen automatically.

    assignments: A dictionary mapping from nengo objects to component indices,
        with 0 as the first component. Used to hard-assign nengo objects
        to specific components.
    """

    if assignments is not None:
        assignments = assignments.copy()
    else:
        assignments = {}

    if num_components == 1:
        return {}

    component_0, G = network_to_filter_graph(network)

    ordering = nx.spectral_ordering(G)

    total_neurons = sum(
        e.n_neurons
        for e in network.all_ensembles
        if not isinstance(e.neurons, Direct))

    component = 0

    if component_0:
        total_neurons -= component_0.n_neurons
        component_0.assign_to_component(assignments, 0)

        ordering.remove(component_0)
        component = 1

    neurons_per_component = float(total_neurons) / num_components

    while ordering:
        next_index = max(
            range(len(ordering)),
            key=lambda i: ordering[i].n_neurons)

        component_n_neurons = 0
        while ordering and component_n_neurons < neurons_per_component:
            ordering[next_index].assign_to_component(
                assignments, component)

            component_n_neurons += ordering[next_index].n_neurons

            del ordering[next_index]

            if next_index >= len(ordering):
                next_index = len(ordering) - 1

        component += 1

    return assignments


def write_metis_input_file(filter_graph, filename):
    """
    Note this this currently relies critically on the order of the nodes
    in the filter graph...but that should be deterministic.
    """

    with open(filename, 'w') as f:
        n = filter_graph.number_of_nodes()
        m = filter_graph.number_of_edges()

        f.write("%d %d 011" % (n, m))

        indices = {node: i+1 for i, node in enumerate(filter_graph.nodes())}

        for u in filter_graph.nodes():
            f.write('\n')

            vertex_weight = 0

            for obj in u.objects:
                if hasattr(obj, 'n_neurons'):
                    vertex_weight += obj.n_neurons

            f.write("%d" % vertex_weight)

            for v, weight_dict in filter_graph[u].iteritems():
                f.write(" %d %d" % (indices[v], weight_dict['weight']))


def read_metis_output_file(filename):
    """
    Read the given file name, assuming it is the output from a run of gpmetis.
    The format of the file is: n lines (n is the number of nodes), the i-th
    line giving information about the i-th node. Each line contains a single
    int giving the component that the i-th node is assigned to. Components are
    indexed starting from 0.
    """

    node_assignments = []

    with open(filename, 'r') as f:
        for line in iter(f.readline, ''):
            node_assignments.append(int(line))

    return node_assignments


def metis_input_filename(network):
    file_name = str(network).replace(' ', '_')
    file_name = file_name.replace('>', '')
    file_name = file_name.replace('<', '')
    file_name = file_name.replace('\"', '')
    file_name = file_name.replace('\'', '')
    return file_name + ".metis"


def metis_output_filename(network, num_components):
    return (
        metis_input_filename(network) + ".part." + str(num_components))


def metis_partitioner(network, num_components, assignments=None):
    """
    If a file with the correct name exists, load that file instead of
    creating a new one.
    """

    if assignments is not None:
        assignments = assignments.copy()
    else:
        assignments = {}

    if num_components == 1:
        return {}

    component_0, G = network_to_filter_graph(network)

    file_name = metis_input_filename(network)

    if not os.path.isfile(file_name):
        print "Writing metis file: %s" % file_name
        write_metis_input_file(G, file_name)

    print "Launching metis..."
    # launch metis with written file
    subprocess.check_call(['gpmetis', file_name, str(num_components)])

    file_name = metis_output_filename(network, num_components)

    print "Reading metis output file: %s" % file_name
    node_assignments = read_metis_output_file(file_name)

    for node, component in zip(G.nodes(), node_assignments):
        for obj in node.objects:
            assignments[obj] = component

    return assignments

class Partitioner(object):
    """
    A class for dividing a nengo network into components.

    Parameters
    ----------
    num_components: The number of components to divide the nengo network into.
        If None, defaults to 1, and ``assignments'' and ``func'' are ignored.

    assignments: A dictionary mapping from nengo objects to component indices,
        with 0 as the first component. Used to hard-assign nengo objects
        to specific components.

    func: A function to partition the nengo graph, assigning nengo objects
        to component indices.

        Arguments:
            network
            num_components
            assignments

    args: Extra positional args passed to func

    kwargs: Extra keyword args passed to func

    """

    def __init__(self, num_components=None, assignments=None,
                 func=None, *args, **kwargs):

        if num_components is None:
            self.num_components = 1

            if func is not None:
                warnings.warn(
                    "Number of components not specified. Defaulting to "
                    "1 and ignoring supplied partitioner function.")

            if assignments is not None:
                warnings.warn(
                    "Number of components not specified. Defaulting to "
                    "1 and ignoring supplied assignments.")

            self.func = self.default_partition_func
            self.assignments = {}

        else:
            self.num_components = num_components

            if func is None:
                func = self.default_partition_func

            self.func = func

            if assignments is None:
                assignments = {}
            else:
                assert isinstance(assignments, dict)

                max_component = max(assignments.values()) if assignments else 0

                if not max_component < num_components:
                    raise ValueError(
                        "``assignments'' dictionary supplied to "
                        "``Partitioner'' requires more components "
                        "than specified by ``num_components''.")

                for k, v in assignments.iteritems():
                    assignments[k] = int(v)

            self.assignments = assignments

        self.args = args
        self.kwargs = kwargs

    @property
    def default_partition_func(self):
        return spectral_partitioner
        #  return metis_partitioner

    def partition(self, network):
        """
        Parameters
        ----------
        network: The network to partition.

        Returns
        -------
        num_components: The number of components the nengo network
            is split into.

        assignments: A dictionary mapping from nengo objects to
            component indices.

        """
        assignments = self.func(
            network, self.num_components, self.assignments,
            *self.args, **self.kwargs)

        propogate_assignments(network, assignments)

        if self.num_components > 1:
            evaluate_partition(network, self.num_components, assignments)

        return self.num_components, assignments


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
    objects are assigned to the master component (component 0).

    These objects are:
        1. Nodes with callable outputs.
        2. Ensembles of Direct neurons.
        3. Any node that is the source of a Connection that has a function.

    Parameters
    ----------
    network: The network we are partitioning.

    assignments: A dictionary mapping from nengo objects to component indices.
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
        outputs: a dict mapping each nengo objects to its output connections.
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
        raise ValueError("Invalid Partition. Error: " + e.message)

    nodes = network.all_nodes
    nodes_in = all([node in assignments for node in nodes])
    assert nodes_in, "Assignments incomplete, missing nodes."

    ensembles = network.all_ensembles
    ensembles_in = all([ensemble in assignments for ensemble in ensembles])
    assert ensembles_in, "Assignments incomplete, missing ensembles."

    probes = network.all_probes
    probes_in = all([probe in assignments for probe in probes])
    assert probes_in, "Assignments incomplete, missing probes."


def evaluate_partition(
        network, num_components, assignments, filter_graph=None):

    print "*" * 80

    if filter_graph is None:
        _, filter_graph = network_to_filter_graph(network)

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

    component_neuron_counts = [0] * num_components
    component_item_counts = [0] * num_components

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

    print "Total number of neurons: ", count_neurons(network)[network]
    print "Mean neurons per cluster: ", mean_neuron_count
    print "Standard deviation of neurons per cluster", neuron_count_std
    print "Min number of neurons", np.min(component_neuron_counts)
    print "Max number of neurons", np.max(component_neuron_counts)
    print (
        "Number of empty partitions",
        num_components - np.count_nonzero(component_neuron_counts))

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
