"""
Contains EnsembleArraySplitter, a class that can be used to split nengo
ensemble arrays up into a larger number of smaller ensemble arrays
that are functionally equivalent. This can create a larger number of
independently-simulatable components for nengo_mpi, allowing a larger
number of processors to be used.
"""


from __future__ import absolute_import
import collections
import numpy as np
import logging

import nengo
from nengo.utils.builder import full_transform, remove_passthrough_nodes


def find_all_io(connections):
    """Build up a list of all inputs and outputs for each object"""

    inputs = collections.defaultdict(list)
    outputs = collections.defaultdict(list)
    for c in connections:
        inputs[c.post_obj].append(c)
        outputs[c.pre_obj].append(c)
    return inputs, outputs


def remove_from_network(network, obj):
    """Remove ``obj'' from network.

    Returns True if ``obj'' was successfully found and removed."""

    if obj in network.objects[type(obj)]:
        network.objects[type(obj)].remove(obj)
        return True

    for sub_net in network.networks:
        removed = remove_from_network(sub_net, obj)

        if removed:
            return True

    return False


def remove_all_connections(network, ea=False):
    """Remove all connections from a network. 

    ``ea'' controls whether to remove from EnsembleArrays"""

    if isinstance(network, nengo.networks.EnsembleArray) and not ea:
        return

    l = []
    network.objects[nengo.Connection] = l
    network.connections = l

    for n in network.networks:
        remove_all_connections(n, ea)


def objs_connections_ensemble_arrays(network):
    """Given a Network, returns (objs, conns, eas).

    Where ``objs '' is a list of (almost) all ensembles and nodes in the
    network, ``conns'' is a list of all (almost) all connections in the
    network, and eas is a list of all ensemble arrays in the network. Note
    that objs and conns do not contain objects that are contained within
    ensemble arrays.
    """

    objs = list(network.ensembles + network.nodes)
    connections = list(network.connections)
    e_arrays = []

    for subnetwork in network.networks:
        if isinstance(subnetwork, nengo.networks.EnsembleArray):
            e_arrays.append(subnetwork)
        else:
            sub_objs, sub_conns, sub_e_arrays = (
                objs_connections_ensemble_arrays(subnetwork))

            objs.extend(sub_objs)
            connections.extend(sub_conns)
            e_arrays.extend(sub_e_arrays)

    return objs, connections, e_arrays


def find_object_location(network, obj):
    """
    Returns a list of the the parent networks of ``obj'' in ``network'',
    sorted in order of increasing specificity.

    Returns an empty list if ``obj'' not found in ``network'' at any level.
    """

    cls = filter(
        lambda cls: cls in network.objects, obj.__class__.__mro__)

    if cls and obj in network.objects[cls[0]]:
        return [network]

    for sub_net in network.networks:
        path = find_object_location(sub_net, obj)

        if path:
            return [network] + path

    return []


def label_or(o, f):
    return o.label if o.label else str(f(o))


def hierarchical_labelling(network, prefix="", delim="."):
    """ Prepend the label of every object in the network with a string giving
    all the objects parent networks. Intended for debugging purposes."""

    class_name = network.__class__.__name__
    prefix += "<%s: %s>%s" % (class_name, label_or(network, id), delim)

    for e in network.ensembles:
        e.label = prefix + label_or(e, id)

    for n in network.nodes:
        n.label = prefix + label_or(n, id)

    for p in network.probes:
        p.label = prefix + label_or(p, id)

    for n in network.networks:
        hierarchical_labelling(n, prefix, delim)
        n.label = prefix + label_or(n, id)


class EnsembleArraySplitter(object):
    """
    After calling EnsembleArraySplitter.split(m, max_neurons), the nengo
    network ``m'' will be modified so that none of its ensemble arrays contain
    more than ``max_neurons' neurons.
    """

    def __init__(self):
        pass

    def split(self, network, max_neurons, preserve_zero_conns=False):
        self.top_level_network = network

        self.log_file_name = 'ensemble_array_splitter.log'
        self.logger = logging.getLogger("split_ea")
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.FileHandler(
            filename=self.log_file_name, mode='w'))

        self.max_neurons = max_neurons
        self.preserve_zero_conns = preserve_zero_conns

        self.node_map = collections.defaultdict(list)

        self.logger.info("\nRelabelling network hierarchically.")
        hierarchical_labelling(network)

        self.logger.info("\nRemoving passthrough nodes.")
        objs, conns, e_arrays = objs_connections_ensemble_arrays(network)
        objs, conns, removed_objs = remove_passthrough_nodes(objs, conns)

        self.logger.info("\nRemoving nodes:")
        for obj in removed_objs:
            assert remove_from_network(network, obj)
            self.logger.info(obj)

        removed_objs = set(removed_objs)

        self.logger.info(
            "\nRemoving probes because their targets have been removed: %s")
        for p in network.all_probes:
            if p.target in removed_objs:
                remove_from_network(network, p)
                self.logger.info(p)

        self.logger.info(
            "\nReplacing connections. "
            "All connections after removing connections:")
        remove_all_connections(network, ea=False)
        for conn in network.all_connections:
            self.logger.info(conn)

        self.logger.info("\nAdding altered connections.")

        network.connections.extend(conns)

        self.inputs, self.outputs = find_all_io(network.all_connections)

        self.logger.info("\n" + "*" * 20 + "Beginning split process" + "*" * 20)
        self.split_helper(network)

        self.probe_map = collections.defaultdict(list)

        for node in self.node_map:
            probes_targeting_node = filter(
                lambda p: p.target is node, network.all_probes)

            for probe in probes_targeting_node:
                assert remove_from_network(network, probe)

                # Add new probes for that node
                for i, n in enumerate(self.traverse_node_map(node)):
                    with network:
                        p = nengo.Probe(
                            n, label="%s_%d" % (probe.label, i),
                            synapse=probe.synapse,
                            sample_every=probe.sample_every,
                            seed=probe.seed, solver=probe.solver)

                        self.probe_map[probe].append(p)

        self.logger.handlers[0].close()
        self.logger.removeHandler(self.logger.handlers[0])

    def traverse_node_map(self, node):
        mapped_nodes = []

        if node in self.node_map:
            for n in self.node_map[node]:
                if n in self.node_map:
                    mapped_nodes += self.traverse_node_map(n)
                else:
                    mapped_nodes.append(n)

        return mapped_nodes

    def split_helper(self, network):
        self.logger.info("In split_helper with %s", network)

        for net in network.networks:
            if isinstance(net, nengo.networks.EnsembleArray):
                n_neurons = sum([e.n_neurons for e in net.all_ensembles])
                n_parts = int(np.ceil(float(n_neurons) / self.max_neurons))
                n_parts = min(n_parts, net.n_ensembles)
                self.split_ensemble_array(net, network, n_parts)
            else:
                self.split_helper(net)

    def split_ensemble_array(self, array, parent, n_parts):
        """
        Splits an ensemble array into multiple functionally equivalent ensemble
        arrays, removing old connections and probes and adding new ones.

        Parameters
        ----------
        array: nengo.EnsembleArray
            The array to split

        parent: nengo.Network
            The network that ``array'' is contained in

        n_parts: int
            Number of arrays to split ``array'' into
        """

        if hasattr(array, 'neuron_input') or hasattr(array, 'neuron_output'):
            self.logger.info(
                "Not splitting ensemble array " + array.label +
                " because it has neuron nodes.")
            return

        if n_parts < 2:
            self.logger.info(
                "Not splitting ensemble array because the "
                "desired number of parts is < 2.")
            return

        self.logger.info("+" * 80)
        self.logger.info(
            "Splitting ensemble array %s into %d parts.",
            array.__repr__(), n_parts)

        if not isinstance(array, nengo.networks.EnsembleArray):
            raise ValueError("'array' must be an EnsembleArray")
        if (not isinstance(parent, nengo.Network)
                or array not in parent.networks):
            raise ValueError("'parent' must be parent network")

        inputs, outputs = self.inputs, self.outputs

        n_ensembles = array.n_ensembles
        D = array.dimensions_per_ensemble

        if array.n_ensembles != len(array.ea_ensembles):
            raise ValueError("Number of ensembles does not match")

        # assert no extra connections
        ea_ensemble_set = set(array.ea_ensembles)
        if len(outputs[array.input]) != n_ensembles or (
                set(c.post for c in outputs[array.input]) != ea_ensemble_set):
            raise ValueError("Extra connections from array input")

        output_nodes = [n for n in array.nodes if n.label[-5:] != 'input']
        assert len(output_nodes) > 0
        # assert len(filter(lambda x: x.label == 'output', output_nodes)) > 0

        for output_node in output_nodes:
            if len(inputs[output_node]) != n_ensembles or (
                    set(c.pre for c in inputs[output_node]) != ea_ensemble_set):
                raise ValueError("Extra connections to array output")

        # equally distribute ensembles between partitions
        sizes = np.zeros(n_parts, dtype=int)
        j = 0
        for i in range(n_ensembles):
            sizes[j] += 1
            j = (j + 1) % len(sizes)

        indices = np.zeros(len(sizes) + 1, dtype=int)
        indices[1:] = np.cumsum(sizes)

        self.logger.info("*" * 10 + "Fixing input connections")

        # make new input nodes
        with array:
            new_inputs = [nengo.Node(size_in=size * D,
                                     label="%s%d" % (array.input.label, i))
                          for i, size in enumerate(sizes)]

        self.node_map[array.input] = new_inputs

        # remove connections involving old input node
        for conn in array.connections[:]:
            if conn.pre_obj is array.input and conn.post in array.ea_ensembles:
                array.connections.remove(conn)

        # make connections from new input nodes to ensembles
        for i, inp in enumerate(new_inputs):
            i0, i1 = indices[i], indices[i+1]
            for j, ens in enumerate(array.ea_ensembles[i0:i1]):
                with array:
                    nengo.Connection(inp[j*D:(j+1)*D], ens, synapse=None)

        # make connections into EnsembleArray
        for c_in in inputs[array.input]:

            # remove connection to old node
            self.logger.info("Removing connection from network: %s", c_in)

            pre_outputs = outputs[c_in.pre_obj]

            transform = full_transform(
                c_in, slice_pre=False, slice_post=True, allow_scalars=False)

            # make connections to new nodes
            for i, inp in enumerate(new_inputs):
                i0, i1 = indices[i], indices[i+1]
                sub_transform = transform[i0*D:i1*D, :]

                if self.preserve_zero_conns or np.any(sub_transform):
                    containing_network =  find_object_location(
                        self.top_level_network, c_in)

                    if not containing_network:
                        print "c_in: " , c_in
                        print "all_connection:"
                        for c in self.top_level_network.all_connections:
                            print c


                    with containing_network[-1]:
                        new_conn = nengo.Connection(
                            c_in.pre, inp,
                            synapse=c_in.synapse,
                            function=c_in.function,
                            transform=sub_transform)

                    self.logger.info("Added connection: %s", new_conn)

                    inputs[inp].append(new_conn)
                    pre_outputs.append(new_conn)

            assert remove_from_network(self.top_level_network, c_in)
            pre_outputs.remove(c_in)

        # remove old input node
        array.nodes.remove(array.input)
        array.input = None

        self.logger.info("*" * 10 + "Fixing output connections")

        # loop over outputs
        for old_output in output_nodes:

            output_sizes = []
            for ensemble in array.ensembles:
                conn = filter(
                    lambda c: old_output.label in str(c.post),
                    outputs[ensemble])[0]
                output_sizes.append(conn.size_out)

            # make new output nodes
            new_outputs = []
            for i in range(n_parts):
                i0, i1 = indices[i], indices[i+1]
                i_sizes = output_sizes[i0:i1]
                with array:
                    new_output = nengo.Node(
                        size_in=sum(i_sizes),
                        label="%s_%d" % (old_output.label, i))

                new_outputs.append(new_output)

                i_inds = np.zeros(len(i_sizes) + 1, dtype=int)
                i_inds[1:] = np.cumsum(i_sizes)

                # connect ensembles to new output node
                for j, e in enumerate(array.ea_ensembles[i0:i1]):
                    old_conns = [c for c in array.connections
                                 if c.pre is e and c.post_obj is old_output]
                    assert len(old_conns) == 1
                    old_conn = old_conns[0]

                    # remove old connection from ensembles
                    array.connections.remove(old_conn)

                    # add new connection from ensemble
                    j0, j1 = i_inds[j], i_inds[j+1]
                    with array:
                        nengo.Connection(
                            e, new_output[j0:j1],
                            synapse=old_conn.synapse,
                            function=old_conn.function,
                            transform=old_conn.transform)

            self.node_map[old_output] = new_outputs

            # connect new outputs to external model
            output_sizes = [n.size_out for n in new_outputs]
            output_inds = np.zeros(len(output_sizes) + 1, dtype=int)
            output_inds[1:] = np.cumsum(output_sizes)

            for c_out in outputs[old_output]:
                assert c_out.function is None

                # remove connection to old node
                self.logger.info("Removing connection from network: %s", c_out)

                transform = full_transform(
                    c_out, slice_pre=True, slice_post=True,
                    allow_scalars=False)

                post_inputs = inputs[c_out.post_obj]

                # add connections to new nodes
                for i, out in enumerate(new_outputs):
                    i0, i1 = output_inds[i], output_inds[i+1]
                    sub_transform = transform[:, i0:i1]

                    if self.preserve_zero_conns or np.any(sub_transform):
                        with find_object_location(
                                self.top_level_network, c_out)[-1]:

                            new_conn = nengo.Connection(
                                out, c_out.post,
                                synapse=c_out.synapse,
                                transform=sub_transform)

                        self.logger.info("Added connection: %s", new_conn)

                        outputs[out].append(new_conn)
                        post_inputs.append(new_conn)

                assert remove_from_network(self.top_level_network, c_out)
                post_inputs.remove(c_out)

            # remove old output node
            array.nodes.remove(old_output)
            setattr(array, old_output.label, None)

    def unsplit_data(self, simulator):
        """
        When ``split'' is called on a network, we sometimes have to split a
        Node up into multiple Nodes. If the original Node was probed, then
        each of the new Nodes are probed as well. This function returns an
        object similar to ``sim.data'' (where ``sim'' is an instance of nengo
        Simulator), but modified to contain the same keys as if ``split'' were
        never called.
        """

        new_data = {}

        handled_ids = []

        for probe_id, p_ids in self.probe_map.iteritems():
            new_data[probe_id] = np.concatenate(
                [simulator.data[i] for i in p_ids], axis=1)

            handled_ids.extend(p_ids)

        remaining_ids = set(simulator.data.keys()) - set(handled_ids)
        remaining_ids = list(remaining_ids)

        for p_id in remaining_ids:
            assert p_id not in new_data
            new_data[p_id] = simulator.data[p_id]

        return new_data
