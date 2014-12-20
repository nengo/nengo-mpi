import random

from nengo import builder
from nengo.ensemble import Ensemble
from nengo.node import Node
from nengo.connection import Connection

import logging
logger = logging.getLogger(__name__)


def nice_str(lst):
    return '[%s]' % '\n '.join(map(str, lst))


def random_partitioner(network, num_components, assignments=None):
    """
    Partition functions must return a dictionary which contains every
    ensemble and every node in the entire network, including subnetworks.

    Puts all nengo Nodes on partition 0.

    Pseudo-randomly assigns nengo Ensembles to remaining partition components,
    unless an Ensemble appears as a key in assignments, in which case the
    integer that it maps to will be used as the component for that Ensemble.

    Parameters
    ----------
    network: The nengo network to partition.

    num_components: The number of components to divide the nengo graph into.
        If None, appropriate value chosen automatically.

    assignments: A dictionary mapping from nengo objects to component indices,
        with 0 as the first component. Used to hard-assign nengo objects
        to specific components.
    """

    if assignments is None:
        assignments = {}
    else:
        assignments = assignments.copy()

    for ensemble in network.ensembles:
        if ensemble not in assignments:
            if num_components == 1:
                assignments[ensemble] = 0
            else:
                assignments[ensemble] = (
                    random.choice(xrange(1, num_components)))

    for node in network.nodes:
        if node.output is None:
            if node not in assignments:
                if num_components == 1:
                    assignments[node] = 0
                else:
                    assignments[node] = (
                        random.choice(xrange(1, num_components)))
        else:
            assignments[node] = 0

    for n in network.networks:
        if n in assignments:
            for ensemble in n.all_ensembles:
                assignments[ensemble] = assignments[n]

            for node in n.all_nodes:
                if node.output is None:
                    assignments[node] = assignments[n]
                else:
                    assignments[node] = 0
        else:
            subnet_assignments = random_partitioner(
                n, num_components, assignments)

            assignments.update(subnet_assignments)

    return assignments


class Partitioner(object):
    """
    A class for dividing a nengo network into components.

    Parameters
    ----------
    num_components: The number of components to divide the nengo graph into.
        If None, appropriate value chosen automatically.

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
                # TODO: issue a warning that its being ignored
                pass

            if assignments is not None:
                # TODO: issue a warning that its being ignored
                pass

            self.func = random_partitioner
            self.assignments = {}

        else:
            self.num_components = num_components

            if func is None:
                func = random_partitioner

            self.func = func

            if assignments is None:
                assignments = {}

            self.assignments = assignments

        self.args = args
        self.kwargs = kwargs

    def partition(self, model, network):
        """
        Returns a list of (non-mpi) models, one for each component
        of the partition. Each model contains the operators assigned
        to a single component of the partition, extracted from the
        MpiModel passed in.

        Parameters
        ----------
        model: An instance of MpiModel, built using the ref-impl builder, from
            the nengo Network ``network''.

        network: A nengo Network. Ops and signals implementing this network are
            stored in ``model''.

        """
        assignments = self.func(
            network, self.num_components, self.assignments,
            *self.args, **self.kwargs)

        return apply_partition(
            model, network, self.num_components, assignments)


def split_connection(conn_ops, signal):
    """
    Split the operators belonging to a connection into a
    ``pre'' group and a ``post'' group. The connection is assumed
    to contain exactly 1 operation performing an update, which
    is assigned to the pre group. All ops that write to signals
    which are read by this updating op are assumed to belong to
    the pre group (as are all ops that write to signals which
    *those* ops read from, etc.). The remaining ops are assigned
    to the post group.

    Parameters
    ----------
    conn_ops: A list containing the operators implementing a nengo connection.

    signal: The signal where the connection will be split. Must be updated by
        one of the operators in ``conn_ops''.

    """

    pre_ops = []

    for op in conn_ops:
        if signal in op.updates:
            pre_ops.append(op)

    assert len(pre_ops) == 1

    reads = pre_ops[0].reads

    post_ops = filter(
        lambda op: op not in pre_ops, conn_ops)

    changed = True
    while changed:
        changed = []

        for op in post_ops:
            writes = set(op.incs) | set(op.sets)

            if writes & set(reads):
                pre_ops.append(op)
                reads.extend(op.reads)
                changed.append(op)

        post_ops = filter(
            lambda op: op not in changed, post_ops)

    return pre_ops, post_ops


def apply_partition(model, network, num_components, assignments):
    """
    Returns a list of (non-mpi) models, one for each component
    of the partition. Each model contains the operators assigned
    to a single component of the partition, extracted from the
    MpiModel passed in.

    Parameters
    ----------
    model: An instance of MpiModel, built using the ref-impl builder,
        implementing the nengo Network ``network''.

    network: A nengo Network. Ops and signals implementing this network are
        stored in ``model''.

    num_components: The number of components to divide the network into.

    assignments: A dictionary mapping from objects in the nengo network to
        component indices, with 0 as the first component. Currently, it
        must contain every node and ensemble in the network, but will not
        contain any information about networks.
    """

    # Checks
    nodes = network.all_nodes
    nodes_in = all([node in assignments for node in nodes])
    assert nodes_in, "Assignments incomplete, missing nodes."

    ensembles = network.all_ensembles
    ensembles_in = all([ensemble in assignments for ensemble in ensembles])
    assert ensembles_in, "Assignments incomplete, missing ensembles."

    # Initialize component models
    models = [builder.Model(label="MPI Component %d" % i)
              for i in range(num_components)]

    for m in models:
        m.sig = model.sig

        m.send_signals = []
        m.recv_signals = []

    # Handle nodes and ensembles
    neuron_assignments = {}

    for obj, component in assignments.iteritems():

        assert isinstance(obj, Ensemble) or isinstance(obj, Node)

        if isinstance(obj, Ensemble):
            neuron_assignments[obj.neurons] = component

        m = models[component]

        for op in model.object_ops[obj]:
            m.add_op(op)

    assignments.update(neuron_assignments)

    # Handle probes
    for probe in network.all_probes:

        component = assignments[probe.target]

        for op in model.object_ops[probe]:
            models[component].add_op(op)

        models[component].probes.append(probe)
        assignments[probe] = component

    # Handle connections
    connections = set(
        obj for obj in model.object_ops
        if isinstance(obj, Connection))

    connections = connections | set(network.all_connections)

    for conn in connections:
        pre_component = assignments[conn.pre_obj]
        post_component = assignments[conn.post_obj]

        if pre_component == post_component:
            # conn on a single component
            m = models[pre_component]

            for op in model.object_ops[conn]:
                m.add_op(op)
        else:
            # conn crosses component boundaries
            if conn.modulatory:
                raise Exception(
                    "Connections crossing component boundaries "
                    "must not be modulatory.")

            if 'synapse_out' in model.sig[conn]:
                signal = model.sig[conn]['synapse_out']
            else:
                raise Exception(
                    "Connections crossing component boundaries "
                    "must be filtered so that there is an update.")

            models[pre_component].send_signals.append(
                (signal, post_component))

            models[post_component].recv_signals.append(
                (signal, pre_component))

            pre_ops, post_ops = split_connection(
                model.object_ops[conn], signal)

            for op in pre_ops:
                models[pre_component].add_op(op)

            for op in post_ops:
                models[post_component].add_op(op)

    return models
