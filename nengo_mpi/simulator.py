import nengo
from nengo.connection import Connection
from nengo.ensemble import Ensemble
from nengo.neurons import Direct
from nengo.node import Node
from nengo.probe import Probe
from nengo import builder
from nengo.simulator import ProbeDict
import nengo.utils.numpy as npext

import chunk
import mpi_sim

import numpy as np
import random
from collections import defaultdict

import warnings

import logging
logger = logging.getLogger(__name__)

nengo.log(debug=True, path=None)


def make_builder(base):
    def build_object(model, obj):
        try:
            model.push_object(obj)
        except AttributeError:
            raise ValueError(
                "Must use an instance of MpiModel.")

        base(model, obj)
        model.pop_object(obj)

    return build_object


with warnings.catch_warnings():
    builder.Builder.register(Ensemble)(
        make_builder(builder.build_ensemble))

    builder.Builder.register(Node)(
        make_builder(builder.build_node))

    builder.Builder.register(Connection)(
        make_builder(builder.build_connection))

    builder.Builder.register(Probe)(
        make_builder(builder.build_probe))


class MpiModel(builder.Model):

    def __init__(self, dt=0.001, label=None):
        self._object_context = [None]
        self.object_ops = defaultdict(list)
        super(MpiModel, self).__init__(dt, label)

    def __str__(self):
        return "MpiModel: %s" % self.label

    def push_object(self, object):
        self._object_context.append(object)

    def pop_object(self, object):
        self._object_context.pop()

    def add_op(self, op):
        self.object_ops[self._object_context[-1]].append(op)
        print "Adding op to model"
        print "OP:", op
        print "object:", self._object_context[-1]
        print '\n'

        super(MpiModel, self).add_op(op)

    def partition_ops(self, network, num_partitions, partition):
        """
        Returns a list of (non-mpi) models, one for each partition,
        containing the operators designated for that partition.
        The models will have the appropriate mpi operators added in.

        Partition will not contain any information about networks.
        Only ensembles and nodes. It is a map from ensemble/nodes to
        int giving the partition.
        """

        connections = network.connections
        probes = network.probes

        models = [builder.Model(label="Partition %d" % i)
                  for i in range(num_partitions)]

        for model in models:
            model.sig = self.sig

            model.send_signals = []
            model.recv_signals = []

        print "PARTITION"
        print partition

        for obj, p in partition.iteritems():

            assert isinstance(obj, Ensemble) or isinstance(obj, Node)

            model = models[p]

            print "OBJECT_OPS"
            print obj
            print self.object_ops[obj]

            for op in self.object_ops[obj]:
                model.add_op(op)

        for i, model in enumerate(models):
            print "MODEL: ", i, " ", model.operators

        for probe in probes:
            print "PROBE", probe

            p = partition[probe.target]
            print "partition: ", p
            print "probe ops: ", self.object_ops[probe]

            for op in self.object_ops[probe]:
                models[p].add_op(op)

            models[p].probes.append(probe)

        for conn in connections:
            print "CONNECTION", conn

            pre_partition = partition[conn.pre]
            post_partition = partition[conn.post]

            if pre_partition == post_partition:

                model = models[pre_partition]

                for op in self.object_ops[conn]:
                    model.add_op(op)
            else:

                # This is a connection that spans partitions
                if conn.modulatory:
                    raise Exception(
                        "Connections spanning partitions "
                        "must not be modulatory")

                #if (isinstance(conn.pre_obj, Node) or
                #        (isinstance(conn.pre_obj, Ensemble) and
                #         isinstance(conn.pre_obj.neuron_type, Direct))):

                #    if conn.function is None:
                #        signal = self.sig[conn]['in']
                #    else:
                #        signal = [sig for sig in self.sig[conn]
                #                  if hasattr(sig, 'name') and
                #                  sig.name == "%s.output" % conn.label][0]

                if (isinstance(conn.pre_obj, Ensemble) and
                        'synapse_out' in self.sig[conn]):

                    # Formaly broke the graph up at decoded signal
                    # signal = self.sig[conn]['decoded']

                    # Currently break the graph up at the output
                    # of the synapses
                    signal = self.sig[conn]['synapse_out']

                else:
                    raise Exception(
                        "Connections of this type cannot span partitions")

                models[pre_partition].send_signals.append(
                    (signal, post_partition))

                models[post_partition].recv_signals.append(
                    (signal, pre_partition))

                pre_ops = [op for op in self.object_ops[conn]
                           if signal in op.updates]

                post_ops = filter(
                    lambda op: op not in pre_ops, self.object_ops[conn])

                for op in pre_ops:
                    models[pre_partition].add_op(op)

                for op in post_ops:
                    models[post_partition].add_op(op)

        return models


def default_partition_func(network, num_partitions, fixed_nodes=None):
    """
    Puts all nengo Nodes on partition 0.

    Pseudo-randomly assigns nengo Ensembles to remaining partitions, unless
    those an Ensemble appears as a key in fixed_nodes, in which case the
    integer that it maps to will be used as the partition for that Ensemble.

    Probably a good idea to supply your own partition in fixed_nodes.

    Parameters
    ----------
    network: the nengo network to partition
    num_partitions: number of hardware nodes to use for the simulation, and
        hence the number of partitions created by the partition function.
        If None, appropriate value chosen automatically.
    fixed_nodes: a dictionary mapping from nengo objects to indices of
        partitions/hardware nodes. Used to hard-assign nengo objects
        to specific partitions/hardware nodes.
    """

    if fixed_nodes is None:
        fixed_nodes = {}

    partition = {}
    partition.update(fixed_nodes)

    for ensemble in network.ensembles:

        if ensemble not in partition:
            if num_partitions == 1:
                partition[ensemble] = 0
            else:
                partition[ensemble] = random.choice(xrange(1, num_partitions))

    for node in network.nodes:
        if node not in partition:
            partition[node] = 0

    return partition


class PartitionInfo(object):
    """
    Stores info for creating MPI processes and assigns Nengo nodes to
    those processes.

    num_partitions: number of hardware nodes to use for the simulation, and
        hence the number of partitions created by the partition function.
        If None, appropriate value chosen automatically.

    fixed_nodes: a dictionary mapping from nengo objects to indices of
        partitions/hardware nodes. Used to hard-assign nengo objects
        to specific partitions/hardware nodes.

    func: a function to partition the nengo graph, assigning nengo objects
        to hardware nodes.

        Arguments:
            network
            num_partitions
            fixed_nodes

    args: extra positional args passed to func

    kwargs: extra keyword args passed to func
    """

    def __init__(self, num_partitions=None, func=None,
                 fixed_nodes=None, *args, **kwargs):

        if num_partitions is None:
            self.num_partitions = 1

            if func is not None:
                # issue a warning that its being ignored
                pass

            if fixed_nodes is not None:
                # issue a warning that its being ignored
                pass

            self.func = default_partition_func
            self.fixed_nodes = {}

        else:
            self.num_partitions = num_partitions

            if func is None:
                func = default_partition_func

            self.func = func

            if fixed_nodes is None:
                fixed_nodes = {}

            self.fixed_nodes = fixed_nodes

        self.args = args
        self.kwargs = kwargs

    def partition(self, network):
        return self.func(
            network, self.num_partitions, self.fixed_nodes,
            *self.args, **self.kwargs)


class Simulator(object):
    """MPI simulator for nengo 2.0."""

    def __init__(self, network, dt=0.001, seed=None, model=None,
                 partition_info=None):
        """
        Initialize the simulator with a network and (optionally) a model.

        Most of the time, you will pass in a network and sometimes a dt::

            sim1 = nengo.Simulator(my_network)  # Uses default 0.001s dt
            sim2 = nengo.Simulator(my_network, dt=0.01)  # Uses 0.01s dt

        For more advanced use cases, you can initialize the model yourself,
        and also pass in a network that will be built into the same model
        that you pass in::

            sim = nengo.Simulator(my_network, model=my_model)

        If you want full control over the build process, then you can build
        your network into the model manually. If you do this, then you must
        explicitly pass in ``None`` for the network::

            sim = nengo.Simulator(None, model=my_model)

        Parameters
        ----------
        network : nengo.Network instance or None
            A network object to the built and then simulated.
            If a fully built ``model`` is passed in, then you can skip
            building the network by passing in network=None.
        dt : float
            The length of a simulator timestep, in seconds.
        seed : int
            A seed for all stochastic operators used in this simulator.
            Note that there are not stochastic operators implemented
            currently, so this parameters does nothing.
        model : nengo.builder.Model instance or None
            A model object that contains build artifacts to be simulated.
            Usually the simulator will build this model for you; however,
            if you want to build the network manually, or to inject some
            build artifacts in the Model before building the network,
            then you can pass in a ``nengo.builder.Model`` instance.
        partition_info:
            An instance of PartitionInfo storing information for specifying
            how nodes are assigned to MPI processes.
        """

        # Note: seed is not used right now, but one day...
        assert seed is None, "Simulator seed not yet implemented"
        self.seed = np.random.randint(npext.maxint) if seed is None else seed

        self.n_steps = 0
        self.dt = dt

        self.model = MpiModel()

        # probe -> C++ key (int)
        self.probe_keys = {}

        # probe -> python list
        self._probe_outputs = self.model.params

        self.mpi_sim = mpi_sim.PythonMpiSimulator()

        builder.Builder.build(self.model, network)

        if partition_info is None:
            partition_info = PartitionInfo()

        num_partitions = partition_info.num_partitions
        partition = partition_info.partition(network)

        models = self.model.partition_ops(network, num_partitions, partition)

        for model in models:
            mpi_chunk = self.mpi_sim.add_chunk()

            simulator_chunk = chunk.SimulatorChunk(
                mpi_chunk, model, dt, self.probe_keys, self._probe_outputs)

        self.mpi_sim.finalize()

        self.data = ProbeDict(self._probe_outputs)

    def __str__(self):
        return self.mpi_sim.to_string()

    def run_steps(self, steps):
        """Simulate for the given number of `dt` steps."""
        self.mpi_sim.run_n_steps(steps)

        for probe, probe_key in self.probe_keys.items():
            data = self.mpi_sim.get_probe_data(probe_key, np.empty)

            if probe not in self._probe_outputs:
                self._probe_outputs[probe] = data
            else:
                self._probe_outputs[probe].extend(data)

        self.n_steps += steps

    def step(self):
        """Advance the simulator by `self.dt` seconds."""
        self.run_steps(1)

    def run(self, time_in_seconds):
        """Simulate for the given length of time."""
        steps = int(np.round(float(time_in_seconds) / self.dt))
        self.run_steps(steps)

    def trange(self, dt=None):
        dt = self.dt if dt is None else dt
        n_steps = int(np.ceil(self.n_steps * self.dt / dt))
        return dt * np.arange(0, n_steps)

    def reset(self):
        # TODO: clear probes in _probe_outputs
        pass
