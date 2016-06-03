from __future__ import print_function
"""
Contains EnsembleArraySplitter, a class that can be used to split nengo
ensemble arrays up into a larger number of smaller ensemble arrays
that are functionally equivalent. This can create a larger number of
independently-simulatable components for nengo_mpi, allowing a larger
number of processors to be used.
"""


import collections
import numpy as np
import logging
from six import iteritems

import nengo
from nengo.utils.builder import full_transform
from nengo_mpi.partition.base import find_all_io, remove_from_network


def remove_all_connections(network, ea=False):
    """Remove all connections from a network.

    ``ea`` controls whether to remove from EnsembleArrays"""

    if isinstance(network, nengo.networks.EnsembleArray) and not ea:
        return

    l = []
    network.objects[nengo.Connection] = l
    network.connections = l

    for n in network.networks:
        remove_all_connections(n, ea)


def objs_connections_ensemble_arrays(network):
    """Given a Network, returns (objs, conns, eas).

    Where ``objs`` is a list of (almost) all ensembles and nodes in the
    network, ``conns`` is a list of all (almost) all connections in the
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
    Returns a list of the the parent networks of ``obj`` in ``network``,
    sorted in order of increasing specificity.

    Returns an empty list if ``obj`` not found in ``network`` at any level.
    """

    cls = [c for c in obj.__class__.__mro__ if c in network.objects]
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
    """ Relabel nengo objects hierarchically.

    Prepend the label of every object in the network with a string giving
    all the objects' parent networks. """

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
    """ Split the ensemble arrays in a network.

    After calling EnsembleArraySplitter.split(m, max_neurons), the nengo
    network ``m`` will be modified so that none of its ensemble arrays contain
    more than ``max_neurons`` neurons.

    ``preserve_zero_conns`` controls whether or not to remove connections
    in the post-split network that have transform equal to 0.
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
        self.logger.propagate = False

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

        self.logger.info(
            "\n" + "*" * 20 + "Beginning split process" + "*" * 20)
        self.split_helper(network)

        self.probe_map = collections.defaultdict(list)

        for node in self.node_map:
            probes_targeting_node = [
                p for p in network.all_probes if p.target is node]

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
                self.split_ensemble_array(net, n_parts)
            else:
                self.split_helper(net)

    def split_ensemble_array(self, array, n_parts):
        """
        Splits an ensemble array into multiple functionally equivalent ensemble
        arrays, removing old connections and probes and adding new ones. Currently
        will not split ensemble arrays that have neuron output or input nodes, but
        there is no reason this could not be added in the future.

        Parameters
        ----------
        array: nengo.EnsembleArray
            The array to split

        n_parts: int
            Number of arrays to split ``array`` into
        """

        if array.neuron_input is not None or array.neuron_output is not None:
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

        inputs, outputs = self.inputs, self.outputs

        n_ensembles = array.n_ensembles
        D = array.dimensions_per_ensemble

        if n_ensembles != len(array.ea_ensembles):
            raise ValueError("Number of ensembles does not match")

        if len(array.all_connections) != n_ensembles * len(array.all_nodes):
            raise ValueError("Number of connections incorrect.")

        # assert no extra connections
        ea_ensemble_set = set(array.ea_ensembles)
        if len(outputs[array.input]) != n_ensembles or (
                set(c.post for c in outputs[array.input]) != ea_ensemble_set):
            raise ValueError("Extra connections from array input")

        connection_set = set(array.all_connections)

        extra_inputs = set()
        extra_outputs = set()

        for conn in self.top_level_network.all_connections:

            if conn.pre_obj in ea_ensemble_set and conn not in array.all_connections:
                extra_outputs.add(conn)

            if conn.post_obj in ea_ensemble_set and conn not in array.all_connections:
                extra_inputs.add(conn)

        for conn in extra_inputs:
            self.logger.info("\n" + "*" * 20)
            self.logger.info("Extra input connector: %s", conn.pre_obj)
            self.logger.info("Synapse: %s", conn.synapse)
            self.logger.info("Inputs: ")

            for c in inputs[conn.pre_obj]:
                self.logger.info(c)

            self.logger.info("Outputs: ")

            for c in outputs[conn.pre_obj]:
                self.logger.info(c)

        for conn in extra_outputs:

            self.logger.info("\n" + "*" * 20)
            self.logger.info("Extra output connector: %s", conn.post_obj)
            self.logger.info("Synapse: %s", conn.synapse)
            self.logger.info("Inputs: ")

            for c in inputs[conn.post_obj]:
                self.logger.info(c)

            self.logger.info("Outputs: ")

            for c in outputs[conn.post_obj]:
                self.logger.info(c)

        output_nodes = [n for n in array.nodes if n.label[-5:] != 'input']
        assert len(output_nodes) > 0
        # assert len(filter(lambda x: x.label == 'output', output_nodes)) > 0

        for output_node in output_nodes:
            extra_connections = (
                set(c.pre for c in inputs[output_node]) != ea_ensemble_set)
            if extra_connections:
                raise ValueError("Extra connections to array output")

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
            new_inputs = [
                nengo.Node(size_in=size * D,
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
                    containing_network = find_object_location(
                        self.top_level_network, c_in)[-1]
                    assert containing_network, (
                        "Connection %s is not in network." % c_in)

                    with containing_network:
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
                conn = next(
                    c for c in outputs[ensemble]
                    if old_output.label in str(c.post))
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
                        containing_network = find_object_location(
                            self.top_level_network, c_out)[-1]
                        assert containing_network, (
                            "Connection %s is not in network." % c_out)

                        with containing_network:
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
        When ``split`` is called on a network, we sometimes have to split a
        Node up into multiple Nodes. If the original Node was probed, then
        each of the new Nodes are probed as well. This function returns an
        object similar to ``sim.data`` (where ``sim`` is an instance of nengo
        Simulator), but modified to contain the same keys as if ``split`` were
        never called.
        """

        new_data = {}

        handled_ids = []

        for probe_id, p_ids in iteritems(self.probe_map):
            new_data[probe_id] = np.concatenate(
                [simulator.data[i] for i in p_ids], axis=1)

            handled_ids.extend(p_ids)

        remaining_ids = set(simulator.data.keys()) - set(handled_ids)
        remaining_ids = list(remaining_ids)

        for p_id in remaining_ids:
            assert p_id not in new_data
            new_data[p_id] = simulator.data[p_id]

        return new_data


def _create_replacement_connection(c_in, c_out):
    """Generate a new Connection to replace two through a passthrough Node"""
    assert c_in.post_obj is c_out.pre_obj
    assert c_in.post_obj.output is None

    # determine the filter for the new Connection
    if c_in.synapse is None:
        synapse = c_out.synapse
    elif c_out.synapse is None:
        synapse = c_in.synapse
    else:
        raise NotImplementedError('Cannot merge two filters')
        # Note: the algorithm below is in the right ballpark,
        #  but isn't exactly the same as two low-pass filters
        # filter = c_out.filter + c_in.filter

    function = c_in.function
    if c_out.function is not None:
        raise Exception('Cannot remove a Node with a '
                        'function being computed on it')

    # compute the combined transform
    transform = np.dot(full_transform(c_out), full_transform(c_in))

    # check if the transform is 0 (this happens a lot
    #  with things like identity transforms)
    if np.all(transform == 0):
        return None

    c = nengo.Connection(c_in.pre_obj, c_out.post_obj,
                         synapse=synapse,
                         transform=transform,
                         function=function,
                         add_to_container=False)
    return c


def remove_passthrough_nodes(
        objs, connections,
        create_connection_fn=_create_replacement_connection):
    """
    Returns a version of the model without passthrough Nodes

    NOTE: this was ripped and slightly modified from the main nengo repo.

    For some backends (such as SpiNNaker), it is useful to remove Nodes that
    have 'None' as their output.  These nodes simply sum their inputs and
    use that as their output. These nodes are defined purely for organizational
    purposes and should not affect the behaviour of the model.  For example,
    the 'input' and 'output' Nodes in an EnsembleArray, which are just meant to
    aggregate data.

    Note that removing passthrough nodes can simplify a model and may be useful
    for other backends as well.  For example, an EnsembleArray connected to
    another EnsembleArray with an identity matrix as the transform
    should collapse down to D Connections between the corresponding Ensembles
    inside the EnsembleArrays.

    Parameters
    ----------
    objs : list of Nodes and Ensembles
        All the objects in the model
    connections : list of Connections
        All the Connections in the model

    Returns the objs and connections of the resulting model.  The passthrough
    Nodes will be removed, and the Connections that interact with those Nodes
    will be replaced with equivalent Connections that don't interact with those
    Nodes.
    """

    inputs, outputs = find_all_io(connections)
    result_conn = list(connections)
    result_objs = list(objs)
    removed_objs = []

    # look for passthrough Nodes to remove
    for obj in objs:
        if isinstance(obj, nengo.Node) and obj.output is None:
            input_filtered = [
                i for i in inputs[obj] if i.synapse is not None]
            output_filtered = [
                o for o in outputs[obj] if o.synapse is not None]

            if input_filtered and output_filtered:
                logging.info(
                    "Cannot merge two filtered connections. "
                    "Keeping node %s." % obj)
                logging.info("Filtered input connections:")
                for i in input_filtered:
                    logging.info("%s" % i)
                logging.info("Filtered output connections:")
                for o in output_filtered:
                    logging.info("%s" % o)

                continue

            if any(c_in.pre_obj is obj for c_in in inputs[obj]):
                logging.info(
                    "Cannot remove node with feedback. Keeping node %s." % obj)
                continue

            result_objs.remove(obj)
            removed_objs.append(obj)

            # get rid of the connections to and from this Node
            for c in inputs[obj]:
                result_conn.remove(c)
                outputs[c.pre_obj].remove(c)
            for c in outputs[obj]:
                result_conn.remove(c)
                inputs[c.post_obj].remove(c)

            # replace those connections with equivalent ones
            for c_in in inputs[obj]:

                for c_out in outputs[obj]:
                    c = create_connection_fn(c_in, c_out)
                    if c is not None:
                        result_conn.append(c)
                        # put this in the list, since it might be used
                        # another time through the loop
                        outputs[c.pre_obj].append(c)
                        inputs[c.post_obj].append(c)

    return result_objs, result_conn, removed_objs
