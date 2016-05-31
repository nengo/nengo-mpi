from __future__ import print_function
import numpy as np
import atexit
from functools import partial
import logging
import time

import nengo
from nengo.simulator import ProbeDict
from nengo.cache import get_default_decoder_cache
import nengo.utils.numpy as npext
from nengo.exceptions import SimulatorClosed

from nengo_mpi.model import MpiBuilder, MpiModel
from nengo_mpi.partition import Partitioner, verify_assignments

logger = logging.getLogger(__name__)


class Simulator(nengo.Simulator):
    """MPI simulator for nengo 2.0."""

    # Only one instance of nengo_mpi.Simulator can be unclosed at any time
    _open_simulators = []

    def __init__(
            self, network, dt=0.001, seed=None, model=None,
            partitioner=None, assignments=None, save_file=""):
        """
        A simulator that can be executed in parallel using MPI.

        Parameters
        ----------
        network : nengo.Network
            A network object to be built and then simulated.
        dt : float
            The length of a simulator timestep, in seconds.
        seed : int
            A seed for all stochastic operators used in this simulator.
            Note that there are not stochastic operators implemented
            currently, so this parameters does nothing.
        model : nengo.builder.Model
            A model object that contains build artifacts to be simulated.
            Usually the simulator will build this model for you; however,
            if you want to build the network manually, or to inject some
            build artifacts in the Model before building the network,
            then you can pass in an instance of ``MpiModel'' instance
            or a ``nengo.builder.Model`` instance. If the latter, it
            will be converted into an ``MpiModel''.
        partitioner: Partitioner
            Specifies how to assign nengo objects to MPI processes.
            ``partitioner'' and ``assignment'' cannot both be supplied.
        assignments: dict
            Dictionary mapping from nengo objects to indices of
            partitions components. ``partitioner'' and ``assignment''
            cannot both be supplied.
        save_file: string
            Name of file that will store all data added to the simulator.
            The simulator can later be reconstructed from this file. If
            equal to the empty string, then no file is created.

        """
        print("Building MPI model...")
        then = time.time()

        self.runnable = not save_file

        if self.runnable and self._open_simulators:
            raise RuntimeError(
                "Attempting to create active instance of nengo_mpi.Simulator "
                "while another instance exists that has not been "
                "closed. Call `close` on existing instances before "
                "creating new ones.")

        if partitioner is not None and assignments is not None:
            raise ValueError(
                "Cannot supply both ``assignments'' and ``partitioner'' to "
                "Simulator.__init__.")

        if assignments is not None:
            p = verify_assignments(network, assignments)
        else:
            if partitioner is None:
                partitioner = Partitioner()

            print("Partitioning network...")
            p = partitioner.partition(network)

        self.n_components, self.assignments = p

        dt = float(dt)
        self.model = MpiModel(
            self.n_components, self.assignments, dt=dt,
            label="%s, dt=%f" % (network, dt),
            decoder_cache=get_default_decoder_cache(),
            save_file=save_file)

        MpiBuilder.build(self.model, network)

        self.model.decoder_cache.shrink()

        print("Finalizing build...")
        self.model.finalize_build()

        # probe -> list
        self._probe_outputs = self.model.params

        self.data = ProbeDict(self._probe_outputs)

        if self.runnable:
            self._open_simulators.append(self)

            seed = np.random.randint(npext.maxint) if seed is None else seed
            self.reset(seed=seed)

        print("Build took %f seconds." % (time.time() - then))

    @property
    def native_sim(self):
        if not self.model.runnable:
            raise Exception(
                "Cannot access C++ simulator of MpiModel, MpiModel is "
                "not in a runnable state. Either in save-file mode, "
                "or the MpiModel instance has not been finalized.")

        return self.model.native_sim

    @property
    def n_steps(self):
        """(int) The current time step of the simulator."""
        return self.model.get_value(self.model.step)[0]

    @property
    def time(self):
        """(float) The current time of the simulator."""
        return self.model.get_value(self.model.time)[0]

    @property
    def closed(self):
        return self not in self._open_simulators

    def close(self):
        if self.runnable:
            try:
                self.native_sim.close()
                self._open_simulators.remove(self)
            except ValueError:
                raise RuntimeError(
                    "Attempting to close a runnable instance of "
                    "nengo_mpi.Simulator that has already been closed.")

    def reset(self, seed=None):
        if self.closed:
            raise SimulatorClosed("Cannot reset closed MpiSimulator.")

        if seed is not None:
            self.seed = seed

        if self.runnable:
            self.native_sim.reset(self.seed)
        else:
            raise RuntimeError(
                "Attempting to reset a non-runnable instance of "
                "nengo_mpi.Simulator.")

        for pk in self.model.probe_keys:
            self._probe_outputs[pk] = []

    def run(self, time_in_seconds, progress_bar=True, log_filename=""):
        """ Simulate for the given length of time. """

        steps = int(np.round(float(time_in_seconds) / self.dt))
        self.run_steps(steps, progress_bar, log_filename)

    def run_steps(self, steps, progress_bar=True, log_filename=""):
        """ Simulate for the given number of `dt` steps. """
        print("Running MPI simulation...")
        then = time.time()

        if self.closed:
            raise SimulatorClosed(
                "MpiSimulator cannot run because it is closed.")

        self.native_sim.run_n_steps(steps, progress_bar, log_filename)

        if not log_filename:
            print("Execution complete, gathering probe data...")
            for probe, probe_key in self.model.probe_keys.items():
                data = self.native_sim.get_probe_data(probe_key)

                # The C++ code doesn't always exactly preserve the shape
                true_shape = self.model.sig[probe]['in'].shape
                if data[0].shape != true_shape:
                    data = map(
                        partial(np.reshape, newshape=true_shape), data)

                if probe not in self._probe_outputs:
                    self._probe_outputs[probe] = data
                else:
                    self._probe_outputs[probe].extend(data)

        print("Simulation took %f seconds." % (time.time() - then))

    def step(self):
        """ Advance the simulator by `self.dt` seconds. """
        self.run_steps(1)

    def trange(self, dt=None):
        dt = self.dt if dt is None else dt
        n_steps = int(self.n_steps * (self.dt / dt))
        return dt * np.arange(1, n_steps + 1)

    @staticmethod
    @atexit.register
    def close_simulators():
        for sim in Simulator._open_simulators:
            sim.close()

    @staticmethod
    def all_closed():
        return Simulator._open_simulators == []

# Mark features as unsupported by nengo_mpi.
# See nengo/simulator.py for info on how it is used.
Simulator.unsupported = [
    ('test_ensemble.test_noise_copies_ok*',
     'nengo_mpi does not support FilteredNoise.'),
    ('test_processes.test_brownnoise*',
     'nengo_mpi does not support FilteredNoise.'),
    ('test_node.test_none*',
     'No error if nodes output None.'),
    ('test_node.test_unconnected_node*',
     'nengo_mpi does not support unconnected nodes.'),
    ('test_node.test_set_output*',
     'nengo_mpi does not support unconnected nodes.'),
    ('test_node.test_args*',
     'This test fails for an unknown reason'),
    ('test_neurons.test_izhikevich*',
     'nengo_mpi does not support Izhikevich neurons.'),
    ('test_cache.test_cache_works*',
     'Not set up correctly.'),
    ('test_connection.test_dist_transform',
     'nengo_mpi does not support supplying distributions for transforms.'),
    ('test_simulator.test_warn_on_opensim_gc',
     'Fails for an unknown reason.'),
    ('test_processes.test_seed',
     'nengo_mpi cannot seed processes properly.')]
