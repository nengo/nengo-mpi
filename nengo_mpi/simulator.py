#simulator.py

from nengo import builder
from nengo.builder import Builder
from nengo.simulator import SignalDict, ProbeDict
from nengo.utils.graphs import toposort
from nengo.utils.simulator import operator_depencency_graph

import numpy as np
import mpi_sim

#from __future__ import print_function
#import logging
#logger = logging.getLogger(__name__)

def zerod_to_float(val):
	if isinstance(val, np.ndarray):
		val = val + 0.0
	return val

def make_func(func, t_in, takes_input):
    def f():
        return zerod_to_float(func())
    def ft(t):
        return zerod_to_float(func(t))
    def fit(t,i):
        return zerod_to_float(func(t,i))
    
    if t_in and takes_input:
        return fit
    elif t_in or takes_input:
        return ft
    else:
        return f

class MPIProbeDict(ProbeDict):
    """Map from Probe -> ndarray

    This is more like a view on the dict that the simulator manipulates.
    However, for speed reasons, the simulator uses Python lists,
    and we want to return NumPy arrays. Additionally, this mapping
    is readonly, which is more appropriate for its purpose.
    """

    def __init__(self, raw, mpi_sim, probes):
        self.raw = raw
        self.mpi_sim = mpi_sim
        self.probes = probes

    def update_probes(self):
        """Populate self.raw based on self.mpi_sim"""

        for probe in self.probes:
            data = self.mpi_sim.get_probe_data(id(probe), np.empty)
            #doing it this way, data should be a list of ndarrays (one ndarray for each time step)
            self.raw[probe].extend(data)

class Simulator(object):
    """MPI simulator for models."""

    def __init__(self, model, dt=0.001, seed=None, builder=Builder()):
        # Call the builder to build the model
        self.model = builder(model, dt)
        self.dt = dt

        # Use model seed as simulator seed if the seed is not provided
        # Note: seed is not used right now, but one day...
        self.seed = self.model.seed if seed is None else seed

        # -- map from Signal.base -> ndarray
        self.signals = SignalDict(__time__=np.asarray(0.0, dtype=np.float64))
        for op in self.model.operators:
            op.init_signals(self.signals, self.dt)

        self.dg = operator_depencency_graph(self.model.operators)
        self._step_order = [node for node in toposort(self.dg)
                            if hasattr(node, 'make_step')]
        
        self.n_steps = 0

        self._init_mpi()

        self._probe_outputs = self.model.params
        self.data = MPIProbeDict(self._probe_outputs, self.mpi_sim, self.model.probes)

    def _init_mpi(self):

        self.mpi_sim = mpi_sim.PythonMpiSimulatorChunk(self.dt)

        for sig, numpy_array in self.signals.items():
            #print sig
            #print numpy_array
            if numpy_array.shape == ():
                numpy_array = np.array([numpy_array])
            self.mpi_sim.add_signal(id(sig), numpy_array)

        for op in self._step_order:
            op_type = type(op)

            #print op

            if op_type == builder.Reset:
                self.mpi_sim.create_Reset(id(op.dst), op.value)

            elif op_type == builder.Copy:
                self.mpi_sim.create_Copy(id(op.dst), id(op.src))

            elif op_type == builder.DotInc:
                if op.A.shape == ():
                    self.mpi_sim.create_ScalarDotInc(id(op.A), id(op.X), id(op.Y))
                else:
                    self.mpi_sim.create_DotInc(id(op.A), id(op.X), id(op.Y))

            elif op_type == builder.ProdUpdate:
                if op.A.shape == ():
                    self.mpi_sim.create_ScalarProdUpdate(id(op.A), id(op.X), id(op.B), id(op.Y))
                else:
                    self.mpi_sim.create_ProdUpdate(id(op.A), id(op.X), id(op.B), id(op.Y))

            elif op_type == builder.SimLIF:
                n_neurons = op.nl.n_neurons
                tau_ref = op.nl.tau_ref
                tau_rc = op.nl.tau_rc
                self.mpi_sim.create_SimLIF(n_neurons, 
                        tau_rc, tau_ref, self.dt, id(op.J), id(op.output))

            elif op_type == builder.SimLIFRate:
                n_neurons = op.nl.n_neurons
                tau_ref = op.nl.tau_ref
                tau_rc = op.nl.tau_rc
                self.mpi_sim.create_SimLIFRate(n_neurons, 
                        tau_rc, tau_ref, self.dt, id(op.J), id(op.output))

            elif op_type == builder.SimPyFunc:
                t_in = op.t_in
                fn = op.fn
                x = op.x

                if x is None:
                    self.mpi_sim.create_PyFunc(id(op.output), make_func(fn, t_in, False), t_in)
                else:
                    self.mpi_sim.create_PyFuncWithInput(id(op.output), make_func(fn, t_in, True), t_in, id(x), x)

            else:
                raise NotImplementedError('nengo_mpi cannot handle operator of type ' + str(op_type))

        for probe in self.model.probes:
            period = (1 if probe.sample_every is None else int(probe.sample_every / self.dt))
            self.mpi_sim.create_Probe(id(probe), id(self.model.sig_in[probe]), period)

    def step(self):
        """Advance the simulator by `self.dt` seconds.
        """
        self.run_steps(1)

    def run(self, time_in_seconds):
        """Simulate for the given length of time."""
        steps = int(np.round(float(time_in_seconds) / self.dt))
        self.run_steps(steps)

    def run_steps(self, steps):
        """Simulate for the given number of `dt` steps."""
        self.mpi_sim.run_n_steps(steps)

        self.data.update_probes()

        self.n_steps += steps

    def trange(self, dt=None):
        dt = self.dt if dt is None else dt
        last_t = self.n_steps * self.dt - self.dt
        n_steps = self.n_steps if dt is None else int(
            self.n_steps / (dt / self.dt))
        return np.linspace(0, last_t, n_steps)

