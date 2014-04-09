#simulator.py

from nengo import builder, LIF, LIFRate
from nengo.builder import Builder
from nengo.simulator import SignalDict, ProbeDict
from nengo.utils.graphs import toposort
from nengo.utils.simulator import operator_depencency_graph

import numpy as np
import mpi_sim

#from __future__ import print_function
#import logging
#logger = logging.getLogger(__name__)

def checks(val):
    if isinstance(val, list):
        val = np.array(val)

    if isinstance(val, np.ndarray):
        val = val + 0.0

    return val

def make_func(func, t_in, takes_input):
    def f():
        return checks(func())
    def ft(t):
        return checks(func(t))
    def fit(t,i):
        return checks(func(t,i))

    if t_in and takes_input:
        return fit
    elif t_in or takes_input:
        return ft
    else:
        return f

class Simulator(object):
    """MPI simulator for models."""

    #can supply an init_mpi function, which will initialize the simulator in some
    #way other than assuming the model passed in is a nengo model. good for testing.
    def __init__(self, model, dt=0.001, seed=1, builder=Builder(), init_mpi=None):
        self.dt = dt
        self.n_steps = 0

        self.mpi_sim = mpi_sim.PythonMpiSimulatorChunk(self.dt)

        #C++ key -> ndarray
        self.sig_dict = {}

        #probe -> C++ key
        self.probe_keys = {}

        #probe -> python list
        self._probe_outputs = {}

        if init_mpi is None:
            self.model = builder(model, self.dt)
            self._init_from_model()
        else:
            init_mpi(self)

        self.data = ProbeDict(self._probe_outputs)

    def add_dot_inc(self, A_key, X_key, Y_key):

        A = self.sig_dict[A_key]
        X = self.sig_dict[X_key]

        A_shape = A.shape
        X_shape = X.shape

        if A.ndim > 1 and A_shape[0] > 1 and A_shape[1] > 1:
            #check whether A HAS to be treated as a matrix
            self.mpi_sim.create_DotIncMV(A_key, X_key, Y_key)
        else:
            #if it doesn't, treat it as a vector
            A_scalar = A_shape == () or A_shape == (1,)
            X_scalar = X_shape == () or X_shape == (1,)

            # if one of them is a scalar and the other isn't, make A the scalar
            if X_scalar and not A_scalar:
                self.mpi_sim.create_DotIncVV(X_key, A_key, Y_key)
            else:
                self.mpi_sim.create_DotIncVV(A_key, X_key, Y_key)

    def add_signal(self, key, A):
        A_shape = A.shape
        if A.ndim > 1 and A_shape[0] > 1 and A_shape[1] > 1:
            self.mpi_sim.add_matrix_signal(key, A)
        else:
            A = np.squeeze(A)
            if A.shape == ():
                A = np.array([A])
            self.mpi_sim.add_vector_signal(key, A)

        self.sig_dict[key] = A

    def add_probe(self, probe, signal_key, probe_key=None, sample_every=None, period=1):

        if sample_every is not None:
            period = 1 if sample_every is None else int(sample_every / self.dt)
            
        self._probe_outputs[probe] = []
        self.probe_keys[probe] = id(probe) if probe_key is None else probe_key
        self.mpi_sim.create_Probe(self.probe_keys[probe], signal_key, period)

    def _init_from_model(self):
        self.seed = self.model.seed 
        self.signals = SignalDict(__time__=np.asarray(0.0, dtype=np.float64))

        for op in self.model.operators:
            op.init_signals(self.signals, self.dt)

        self.dg = operator_depencency_graph(self.model.operators)
        self._step_order = [node for node in toposort(self.dg)
                            if hasattr(node, 'make_step')]

        for sig, numpy_array in self.signals.items():
            self.add_signal(id(sig), numpy_array)

        for op in self._step_order:
            op_type = type(op)

            #print op

            if op_type == builder.Reset:
                self.mpi_sim.create_Reset(id(op.dst), op.value)

            elif op_type == builder.Copy:
                self.mpi_sim.create_Copy(id(op.dst), id(op.src))

            elif op_type == builder.DotInc:
                self.add_dot_inc(id(op.A), id(op.X), id(op.Y))

            elif op_type == builder.ProdUpdate:
                self.mpi_sim.create_ProdUpdate(id(op.B), id(op.Y))
                self.add_dot_inc(id(op.A), id(op.X), id(op.Y))

            elif op_type == builder.SimNeurons:
                n_neurons = op.neurons.n_neurons

                if type(op.neurons) is LIF:
                    tau_ref = op.neurons.tau_ref
                    tau_rc = op.neurons.tau_rc
                    self.mpi_sim.create_SimLIF(n_neurons,
                            tau_rc, tau_ref, self.dt, id(op.J), id(op.output))
                elif type(op.neurons) is LIFRate:
                    tau_ref = op.neurons.tau_ref
                    tau_rc = op.neurons.tau_rc
                    self.mpi_sim.create_SimLIFRate(n_neurons,
                            tau_rc, tau_ref, self.dt, id(op.J), id(op.output))
                else:
                    raise NotImplementedError('nengo_mpi cannot handle neurons of type ' + str(type(op.neurons)))

            elif op_type == builder.SimPyFunc:
                t_in = op.t_in
                fn = op.fn
                x = op.x

                if x is None:
                    self.mpi_sim.create_PyFunc(id(op.output), make_func(fn, t_in, False), t_in)
                else:
                    self.mpi_sim.create_PyFuncWithInput(id(op.output), make_func(fn, t_in, True), t_in, id(x), x.value)

            else:
                raise NotImplementedError('nengo_mpi cannot handle operator of type ' + str(op_type))

        self._probe_outputs = self.model.params

        for probe in self.model.probes:
            self.add_probe(probe, id(self.model.sig_in[probe]), sample_every=probe.sample_every)

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

        for probe, probe_key in self.probe_keys.items():
            data = self.mpi_sim.get_probe_data(probe_key, np.empty)
            self._probe_outputs[probe].extend(data)

        self.n_steps += steps

    def trange(self, dt=None):
        dt = self.dt if dt is None else dt
        last_t = self.n_steps * self.dt - self.dt
        n_steps = self.n_steps if dt is None else int(
            self.n_steps / (dt / self.dt))
        return np.linspace(0, last_t, n_steps)

