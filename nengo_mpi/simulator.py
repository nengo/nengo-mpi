import nengo
from nengo.connection import Connection
from nengo.ensemble import Ensemble
from nengo.learning_rules import BCM, Oja, PES
from nengo.network import Network
from nengo.neurons import AdaptiveLIF, AdaptiveLIFRate
from nengo.neurons import LIF, LIFRate, Direct
from nengo.node import Node
from nengo.probe import Probe
from nengo.synapses import Alpha, LinearFilter, Lowpass
import nengo.builder as builder
from nengo.simulator import ProbeDict
import nengo.utils.numpy as npext

import chunk
import mpi_sim

import numpy as np
import random
from collections import defaultdict

import logging
logger = logging.getLogger(__name__)

nengo.log(debug=True, path=None)


class MpiBuilder(builder.Builder):
    builders = {}


def make_builder(base):
    def build_object(obj, model, config):
        print model
        try:
            model.push_object(obj)
        except AttributeError:
            raise ValueError(
                "Must use an instance of MpiModel.")

        base(obj, model, config)
        model.pop_object(obj)

    return build_object

builder.Builder.register_builder(
    make_builder(builder.build_ensemble), Ensemble)

builder.Builder.register_builder(
    make_builder(builder.build_node), Node)

builder.Builder.register_builder(
    make_builder(builder.build_connection), Connection)


# Shouldn't have to do this, shoule be able to inherit registered builders
# from builder.Builder
MpiBuilder.register_builder(builder.build_network, Network)
MpiBuilder.register_builder(builder.build_lifrate, LIFRate)
MpiBuilder.register_builder(builder.build_lif, LIF)
MpiBuilder.register_builder(builder.build_alifrate, AdaptiveLIFRate)
MpiBuilder.register_builder(builder.build_alif, AdaptiveLIF)
MpiBuilder.register_builder(builder.build_probe, Probe)
MpiBuilder.register_builder(builder.build_filter_synapse, LinearFilter)
MpiBuilder.register_builder(builder.build_lowpass_synapse, Lowpass)
MpiBuilder.register_builder(builder.build_alpha_synapse, Alpha)
MpiBuilder.register_builder(builder.build_pes, PES)
MpiBuilder.register_builder(builder.build_bcm, BCM)
MpiBuilder.register_builder(builder.build_bcm, Oja)


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
        print "ADD_OP"
        print self._object_context[-1]
        print op
        super(MpiModel, self).add_op(op)

    def partition_ops(self, num_partitions, partition, connections):
        """
        Returns a list of (non-mpi) models, one for each partition,
        containing the operators appropriate for that partition.
        The models will also have the appropriate mpi operators added in.

        Partition will not contain any information about networks.
        Only ensembles and nodes. Map from ensemble/nodes to int
        giving the partition.

        Connections is the list of connections in the network.
        """

        models = [builder.Model(label="Partition %d" % i)
                  for i in range(num_partitions)]

        for model in models:
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

        for conn in connections:
            pre_partition = partition[conn.pre]
            post_partition = partition[conn.post]

            if pre_partition == post_partition:

                model = models[pre_partition]

                for op in self.object_ops[conn]:
                    model.add_op(op)
            else:
                # Have a connection that spans partitions

                if conn.modulatory:
                    raise Exception(
                        "Connections spanning partitions "
                        "must not be modulatory")

                if (isinstance(conn.pre_obj, Node) or
                        (isinstance(conn.pre_obj, Ensemble) and
                         isinstance(conn.pre_obj.neuron_type, Direct))):

                    if conn.function is None:
                        decoded_signal = self.sig[conn]['in']
                    else:
                        decoded_signal = [sig for sig in self.sig[conn]
                                          if hasattr(sig, 'name') and
                                          sig.name == "%s.output" % conn.label]
                        decoded_signal = decoded_signal[0]

                elif isinstance(conn.pre_obj, Ensemble):
                    # Break the connection at the decoded signal.
                    # The decoded signal has name: conn.label
                    decoded_signal = self.sig[conn]['decoded']
                else:
                    raise Exception(
                        "Connections of this type cannot span partitions")

                models[pre_partition].send_signals.append(
                    (decoded_signal, post_partition))

                models[post_partition].recv_signals.append(
                    (decoded_signal, pre_partition))

                pre_ops = [op for op in self.object_ops[conn]
                           if decoded_signal in op.updates]
                post_ops = filter(
                    lambda op: op not in pre_ops, self.object_ops[conn])

                for op in pre_ops:
                    models[pre_partition].add_op(op)

                for op in post_ops:
                    models[post_partition].add_op(op)

        return models


class Simulator(object):
    """MPI simulator for nengo 2.0."""

    def __init__(self, network=None, dt=0.001, seed=None, model=None,
                 init_func=None, num_partitions=None, partition_func=None,
                 fixed_nodes=None):
        """
        (Mostly copied from docstring for nengo.Simulator)

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
        init_func : function that accepts a Simulator, or None
            This function permits the user to call functions like add_signals,
            add_probes. This is useful for testing individual peices of them
            simulator.
        num_partitions: number of hardware nodes to use for the simulation, and
            hence the number of partitions created by the partition function.
            If None, appropriate value chosen automatically.
        partition_func: a function to partition the nengo graph, assigning
            nengo objects to hardware nodes
        fixed_nodes: a dictionary mapping from nengo objects to indices of
            partitions/hardware nodes. Used to hard-assign nengo objects
            to specific partitions/hardware nodes.
        """

        # Note: seed is not used right now, but one day...
        assert seed is None, "Simulator seed not yet implemented"
        self.seed = np.random.randint(npext.maxint) if seed is None else seed

        self.n_steps = 0
        self.dt = dt

        # probe -> C++ key (int)
        self.probe_keys = {}

        # probe -> python list
        self._probe_outputs = {}

        self.model = model

        self.mpi_sim = mpi_sim.PythonMpiSimulator()

        # TODO: use MpiModel in case where a model is passed in (ie convert the
        # model) to an MpiModel.
        if network is not None:

            if self.model is None:
                self.model = builder.Model(
                    dt=dt, label="%s, dt=%f" % (network.label, dt))

            mpi_model = MpiModel()
            builder.Builder.build(network, model=mpi_model)

            if partition_func is None:
                partition_func = default_partition_func

            if fixed_nodes is None:
                fixed_nodes = {}

            partition, num_partitions = partition_func(
                network, num_partitions, fixed_nodes)

            # now mpi model is populated
            models = mpi_model.partition_ops(
                num_partitions, partition, network.connections)

            for model in models:
                mpi_chunk = self.mpi_sim.add_chunk()
                simulator_chunk = chunk.SimulatorChunk(mpi_chunk, model, dt)

            self.mpi_sim.finalize()

        if init_func is not None:
            init_func(self)

        self.data = ProbeDict(self._probe_outputs)

    def __str__(self):
        return self.mpi_sim.to_string()

    def run_steps(self, steps):
        """Simulate for the given number of `dt` steps."""
        self.mpi_sim.run_n_steps(steps)

        for probe, probe_key in self.probe_keys.items():
            data = self.mpi_sim.get_probe_data(probe_key, np.empty)
            self._probe_outputs[probe].extend(data)

        self.n_steps += steps
        self.signals['__time__'] += steps * self.dt

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


def default_partition_func(network, num_partitions=None, fixed_nodes=None):
    """
    Puts all nengo nodes on partition 0.
    Pseudo-randomly assigns nengo ensembles to partitions.

    So its a good idea to supply your own partition in fixed_nodes.

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

    partition = {}
    partition.update(fixed_nodes)

    if num_partitions is None:
        if fixed_nodes is not None:
            num_partitions = len(set(fixed_nodes.values()))
        else:
            num_partitions = 1

    for ensemble in network.ensembles:
        if ensemble not in partition:
            partition[ensemble] = random.choice(xrange(num_partitions))

    for node in network.nodes:
        if node not in partition:
            partition[node] = 0

    return partition, num_partitions
