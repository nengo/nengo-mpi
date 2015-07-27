"""
Contains EnsembleArraySplitter, a class that can be used to split nengo
ensemble arrays up into smaller a larger number of smaller ensemble arrays
with the same functionality. This can create a larger number of
independently-simulatable components for nengo_mpi, allowing a larger
number of processors to be used.
"""


from __future__ import absolute_import
import collections
import numpy as np
import logging

import nengo

def full_transform(conn, slice_pre=True, slice_post=True, allow_scalars=True):
    """Compute the full transform for a connection.

    Parameters
    ----------
    conn : Connection
        The connection for which to compute the full transform.
    slice_pre : boolean, optional (True)
        Whether to compute the pre slice as part of the transform.
    slice_post : boolean, optional (True)
        Whether to compute the post slice as part of the transform.
    allow_scalars : boolean, optional (True)
        If true (default), will not make scalars into full transforms when
        not using slicing, since these work fine in the reference builder.
        If false, these scalars will be turned into scaled identity matrices.
    """
    transform = conn.transform
    pre_slice = conn.pre_slice if slice_pre else slice(None)
    post_slice = conn.post_slice if slice_post else slice(None)

    if pre_slice == slice(None) and post_slice == slice(None):
        if transform.ndim == 2:
            # transform is already full, so return a copy
            return np.array(transform)
        elif transform.size == 1 and allow_scalars:
            return np.array(transform)

    # Create the new transform matching the pre/post dimensions
    func_size = conn.function_info.size
    size_in = (conn.pre_obj.size_out if func_size is None
               else func_size) if slice_pre else conn.size_mid
    size_out = conn.post_obj.size_in if slice_post else conn.size_out
    new_transform = np.zeros((size_out, size_in))

    if transform.ndim < 2:
        new_transform[np.arange(size_out)[post_slice],
                      np.arange(size_in)[pre_slice]] = transform
        return new_transform
    elif transform.ndim == 2:
        repeated_inds = lambda x: (
            not isinstance(x, slice) and np.unique(x).size != len(x))
        if repeated_inds(pre_slice):
            raise ValueError("Input object selection has repeated indices")
        if repeated_inds(post_slice):
            raise ValueError("Output object selection has repeated indices")

        rows_transform = np.array(new_transform[post_slice])
        rows_transform[:, pre_slice] = transform
        new_transform[post_slice] = rows_transform
        # Note: the above is a little obscure, but we do it so that lists of
        #  indices can specify selections of rows and columns, rather than
        #  just individual items
        return new_transform
    else:
        raise ValueError("Transforms with > 2 dims not supported")


def find_all_io(connections):
    """Build up a list of all inputs and outputs for each object"""
    inputs = collections.defaultdict(list)
    outputs = collections.defaultdict(list)
    for c in connections:
        inputs[c.post_obj].append(c)
        outputs[c.pre_obj].append(c)
    return inputs, outputs


def remove_from_network(network, obj):
    """Returns whether removal was successful"""

    if obj in network.objects[type(obj)]:
        network.objects[type(obj)].remove(obj)
        return True

    for sub_net in network.networks:
        removed = remove_from_network(sub_net, obj)

        if removed:
            return True

    return False


def find_probes(network, target):
    probes = []

    for probe in network.probes:
        if probe.target is target:
            probes.append(probe)

    for subnet in network.networks:
        probes.extend(find_probes(subnet, target))

    return probes


def find_object_location(network, obj):
    cls = filter(
        lambda cls: cls in network.objects, obj.__class__.__mro__)

    if cls and obj in network.objects[cls[0]]:
        return [network]

    for sub_net in network.networks:
        path = find_object_location(sub_net, obj)

        if path:
            return [network] + path

    return []


class EnsembleArraySplitter(object):
    """
    After calling EnsembleArraySplitter(m, max_neurons), the nengo network 'm'
    will be modified so that none of its ensemble arrays contain more than
    'max_neurons' neurons.
    """

    def __init__(self):
        pass

    def split(self, network, max_neurons, preserve_zeros=False):
        self.inputs, self.outputs = find_all_io(network.all_connections)
        self.top_level_network = network

        self.logger = logging.getLogger('EnsembleArraySplitter')
        self.logger.setLevel(logging.INFO)

        fh = logging.FileHandler('ensemble_array_splitter.log', mode='w')
        fh.setLevel(logging.INFO)

        self.logger.addHandler(fh)

        self.max_neurons = max_neurons
        self.preserve_zeros = preserve_zeros

        self.fix_special_networks(self.top_level_network)
        self.fix_labels(self.top_level_network)

        self.node_map = {}
        self.split_helper(self.top_level_network)

        self.probe_map = collections.defaultdict(list)
        for node in self.node_map:
            probes_targeting_node = filter(
                lambda p: p.target is node, self.top_level_network.all_probes)

            for probe in probes_targeting_node:
                assert remove_from_network(self.top_level_network, probe)

                # Add new probes for that node
                for i, n in enumerate(self.node_map[node]):
                    with self.top_level_network:
                        p = nengo.Probe(
                            n, label="%s_%d" % (probe.label, i),
                            synapse=probe.synapse,
                            sample_every=probe.sample_every,
                            seed=probe.seed, solver=probe.solver)

                        self.probe_map[probe].append(p)

        fh.close()

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

    def fix_labels(self, network):
        for net in network.networks:
            if isinstance(net, nengo.networks.EnsembleArray):
                for obj in net.nodes:
                    if net.label:
                        obj.label = "%s_%s" % (net.label, obj.label)
            else:
                self.fix_labels(net)

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
            print (
                "Not splitting ensemble array " + array.label +
                " because it has neuron nodes.")
            return

        if n_parts < 2:
            print (
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

        # assert no extra connections
        if array.n_ensembles != len(array.ea_ensembles):
            raise ValueError("Number of ensembles does not match")

        ea_ensemble_set = set(array.ea_ensembles)
        if len(outputs[array.input]) != n_ensembles or (
                set(c.post for c in outputs[array.input]) != ea_ensemble_set):
            raise ValueError("Extra connections from array input")

        for output in (getattr(array, name) for name in array.output_sizes):
            if len(inputs[output]) != n_ensembles or (
                    set(c.pre for c in inputs[output]) != ea_ensemble_set):
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

                if self.preserve_zeros or np.any(sub_transform):
                    with find_object_location(
                            self.top_level_network, c_in)[-1]:
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
        for output_name in array.output_sizes:
            old_output = getattr(array, output_name)
            output_sizes = array.output_sizes[output_name]

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

                    if self.preserve_zeros or np.any(sub_transform):
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
            setattr(array, output_name, None)

    def fix_special_networks(self, network):
        inputs, outputs = self.inputs, self.outputs

        for net in network.networks[:]:
            if net.label == "Circular Convolution":
                print "Fixing circular convolution"

                ea = net.product.product
                network.networks.append(ea)
                inputs[ea.input] = []

                for input_node in [net.A, net.B]:

                    cc_to_prod = outputs[input_node][0]
                    prod_to_ea = outputs[cc_to_prod.post_obj][0]

                    cc_transform = full_transform(
                        cc_to_prod, True, True, True)
                    prod_transform = full_transform(
                        prod_to_ea, True, True, True)

                    for conn in inputs[input_node]:
                        removed = remove_from_network(
                            self.top_level_network, conn)
                        assert removed

                        outputs[conn.pre_obj].remove(conn)

                        transform = np.dot(
                            cc_transform,
                            full_transform(conn, False, True, True))

                        transform = np.dot(
                            prod_transform, transform)

                        with network:
                            new_conn = nengo.Connection(
                                conn.pre, ea.input,
                                synapse=conn.synapse,
                                function=conn.function,
                                transform=transform)

                        inputs[ea.input].append(new_conn)
                        outputs[conn.pre_obj].append(new_conn)

                cc_out_transform = full_transform(
                    inputs[net.output][0], True, True, True)

                outputs[ea.product] = []

                for conn in outputs[net.output]:
                    assert conn.function is None

                    removed = remove_from_network(self.top_level_network, conn)
                    assert removed

                    inputs[conn.post_obj].remove(conn)

                    transform = np.dot(
                        full_transform(conn, True, True, True),
                        cc_out_transform)

                    with network:
                        new_conn = nengo.Connection(
                            ea.product, conn.post_obj,
                            synapse=conn.synapse,
                            transform=transform)

                    outputs[ea.product].append(new_conn)
                    inputs[conn.post_obj].append(new_conn)

                removed = remove_from_network(self.top_level_network, net)
                assert removed
            else:
                self.fix_special_networks(net)
