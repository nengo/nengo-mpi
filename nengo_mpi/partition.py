from nengo import builder
from nengo.connection import Connection
import warnings

import logging
logger = logging.getLogger(__name__)


def top_level_partitioner(network, num_components, assignments=None):
    """
    A partitioner that puts top level subnetworks on different partitions if it
    can.  Everything inside a top level subnetwork (except nodes) will go on
    the same partition.  Reasonable if your network if split up into
    subnetworks of relatively equal size.

    Ensembles at top level will be handled in the same way. All nodes in the
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
            assignments[network] = component
            component = (component + 1) % num_components

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

            self.func = top_level_partitioner
            self.assignments = {}

        else:
            self.num_components = num_components

            if func is None:
                func = top_level_partitioner

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

        return self.num_components, assignments


def propogate_assignments(network, assignments):
    """
    Propogates the component assignments stored in the dict ``assignments''
    (which only needs to contain assignments for top level networks, nodes and
    ensembles) down to objects that are contained in those top-level objects.
    If assignments is empty, then all objects will be assigned to the 1st
    component, which has index 0. The intent is to have some partitioning
    algorithm determine some of the assignments before this function is called,
    and then this function expands those assignments, respecting hierarchical
    constraints.

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
    def helper(network, assignments):

        for node in network.nodes:
            # Non-callable nodes can go on arbitrary processes, as long as
            # connections coming from those nodes do not have functions
            # associated with them. TODO: enforce this rule.

            if callable(node.output):
                assignments[node] = 0
            elif node not in assignments:
                assignments[node] = assignments[network]

        for ensemble in network.ensembles:
            if ensemble not in assignments:
                assignments[ensemble] = assignments[network]

            assignments[ensemble.neurons] = assignments[ensemble]

        for n in network.networks:
            if n not in assignments:
                assignments[n] = assignments[network]

            helper(n, assignments)

    def probe_helper(network, assignments):
        # TODO: properly handle probes that target connections
        # connections will not be in ``assignments'' at this point.
        for probe in network.probes:
            assignments[probe] = assignments[probe.target]

        for n in network.networks:
            probe_helper(n, assignments)

    assignments[network] = 0

    try:
        helper(network, assignments)
        probe_helper(network, assignments)
    except KeyError:
        # Nengo tests require a value error to be raised in these cases.
        raise ValueError("Invalid Partition")

    nodes = network.all_nodes
    nodes_in = all([node in assignments for node in nodes])
    assert nodes_in, "Assignments incomplete, missing nodes."

    ensembles = network.all_ensembles
    ensembles_in = all([ensemble in assignments for ensemble in ensembles])
    assert ensembles_in, "Assignments incomplete, missing ensembles."

    probes = network.all_probes
    probes_in = all([probe in assignments for probe in probes])
    assert probes_in, "Assignments incomplete, missing probes."


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
        must contain every node and ensemble in the network, including
        those in subnetworks.

    Returns
    -------
    models: A list of models, 1 for each component, containing the operators
        and signals implementing the nengo objects assigned to each component.

    """

    # Initialize component models
    models = [builder.Model(label="MPI Component %d" % i)
              for i in range(num_components)]

    for m in models:
        m.sig = model.sig

        m.send_signals = []
        m.recv_signals = []

    # Handle nodes and ensembles
    for obj in network.all_nodes + network.all_ensembles:

        component = assignments[obj]

        for op in model.object_ops[obj]:
            models[component].add_op(op)

    # Handle probes
    for probe in network.all_probes:

        component = assignments[probe.target]

        for op in model.object_ops[probe]:
            models[component].add_op(op)

        models[component].probes.append(probe)

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

            if conn.learning_rule_type:
                raise Exception(
                    "Connections crossing component boundaries "
                    "must not have learning rules.")

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
