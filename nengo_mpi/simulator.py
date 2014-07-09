from nengo import LIF, LIFRate
import nengo.builder as builder
from nengo.simulator import ProbeDict
from nengo.utils.graphs import toposort
from nengo.utils.simulator import operator_depencency_graph
import nengo.utils.numpy as npext

import numpy as np
import mpi_sim

import logging
logger = logging.getLogger(__name__)

# nengo.log(debug=True, path=None)


def make_key(obj):
    if isinstance(obj, builder.SignalView):
        return id(obj.base)
    else:
        return id(obj)


def checks(val):
    """
    If the output can possibly be treated as a scalar, convert it
    to a python float. Otherwise, convert it to a numpy ndarray.
    """

    if isinstance(val, list):
        val = np.array(val, dtype=np.float64)

    elif isinstance(val, np.ndarray):

        if getattr(val, 'shape', None) == ():
            val = float(val)

        elif getattr(val, 'dtype', None) != np.float64:
            val = array(val, dtype=np.float64)

    elif isinstance(val, int):
        val = float(val)

    elif isinstance(val, float):
        if isinstance(val, np.float64):
            val = float(val)

    else:
        raise ValueError("python function returning unexpected value, %s" % str(val))

    return val


def make_checked_func(func, t_in, takes_input):
    def f():
        return checks(func())

    def ft(t):
        return checks(func(t))

    def fit(t, i):
        return checks(func(t, i))

    if t_in and takes_input:
        return fit
    elif t_in or takes_input:
        return ft
    else:
        return f


class Simulator(object):
    """MPI simulator for nengo 2.0."""

    def __init__(self, network, dt=0.001, seed=None, model=None, init_func=None):
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
        """

        # Note: seed is not used right now, but one day...
        assert seed is None, "Simulator seed not yet implemented"
        self.seed = np.random.randint(npext.maxint) if seed is None else seed

        self.n_steps = 0
        self.dt = dt

        # C++ key -> ndarray
        self.sig_dict = {}

        # probe -> C++ key
        self.probe_keys = {}

        # probe -> python list
        self._probe_outputs = {}

        self.model = model
        self.signals = builder.SignalDict(__time__=np.asarray(0.0, dtype=np.float64))

        if network is not None:

            if self.model is None:
                self.model = builder.Model(
                    dt=dt, label="%s, dt=%f" % (network.label, dt))

            builder.Builder.build(network, model=self.model)

        self.mpi_sim = mpi_sim.PythonMpiSimulatorChunk(self.dt)

        if self.model is not None:
            self._init_from_model()

        if init_func is not None:
            init_func(self)

        self.data = ProbeDict(self._probe_outputs)

    def add_dot_inc(self, A_key, X_key, Y_key):

        A = self.sig_dict[A_key]
        X = self.sig_dict[X_key]

        A_shape = A.shape
        X_shape = X.shape

        if A.ndim > 1 and A_shape[0] > 1 and A_shape[1] > 1:
            # check whether A has to be treated as a matrix
            self.mpi_sim.create_DotIncMV(A_key, X_key, Y_key)
            logger.debug(
                "Creating DotIncMV, A:%d, X:%d, Y:%d", A_key, X_key, Y_key)
        else:
            # if it doesn't, treat it as a vector
            A_scalar = A_shape == () or A_shape == (1,)
            X_scalar = X_shape == () or X_shape == (1,)

            # if one of them is a scalar and the other isn't, make A the scalar
            if X_scalar and not A_scalar:
                self.mpi_sim.create_DotIncVV(X_key, A_key, Y_key)
                logger.debug(
                    "Creating DotIncVV(inv), A:%d, X:%d, Y:%d",
                    A_key, X_key, Y_key)
            else:
                logger.debug(
                    "Creating DotIncVV, A:%d, X:%d, Y:%d", A_key, X_key, Y_key)
                self.mpi_sim.create_DotIncVV(A_key, X_key, Y_key)

    def add_signal(self, key, A, label=''):

        A_shape = A.shape

        if A.ndim > 1 and A_shape[0] > 1 and A_shape[1] > 1:
            logger.debug(
                "Creating matrix signal, name: %s, key: %d", label, key)

            self.mpi_sim.add_matrix_signal(key, A)
        else:
            A = np.squeeze(A)

            if A.shape == ():
                A = np.array([A])

            logger.debug(
                "Creating vector signal, name: %s, key: %d", label, key)

            self.mpi_sim.add_vector_signal(key, A)

        self.sig_dict[key] = A

    def add_probe(self, probe, signal_key, probe_key=None,
                  sample_every=None, period=1):

        if sample_every is not None:
            period = 1 if sample_every is None else int(sample_every / self.dt)

        self._probe_outputs[probe] = []
        self.probe_keys[probe] = make_key(probe) if probe_key is None else probe_key
        self.mpi_sim.create_Probe(self.probe_keys[probe], signal_key, period)

    def _init_from_model(self):
        """
        Only to be called if self.model is defined.

        self.mpi_sim must have been created by the time this function is called.
        """

        assert hasattr(self, 'model') and self.model is not None

        for op in self.model.operators:
            op.init_signals(self.signals, self.dt)

        self.dg = operator_depencency_graph(self.model.operators)
        self._step_order = [node for node in toposort(self.dg)
                            if hasattr(node, 'make_step')]

        for sig, numpy_array in self.signals.items():
            self.add_signal(make_key(sig), numpy_array, str(sig))

        for op in self._step_order:
            op_type = type(op)

            if op_type == builder.Reset:
                logger.debug(
                    "Creating Reset, dst:%d, Val:%f", make_key(op.dst), op.value)
                self.mpi_sim.create_Reset(make_key(op.dst), op.value)

            elif op_type == builder.Copy:
                logger.debug(
                    "Creating Copy, dst:%d, src:%d", make_key(op.dst), make_key(op.src))
                self.mpi_sim.create_Copy(make_key(op.dst), make_key(op.src))

            elif op_type == builder.DotInc:
                self.add_dot_inc(make_key(op.A), make_key(op.X), make_key(op.Y))

            elif op_type == builder.ProdUpdate:
                logger.debug(
                    "Creating ProdUpdate, A: %d, X: %d, B:%d, Y:%d",
                    make_key(op.A), make_key(op.X), make_key(op.B), make_key(op.Y))

                self.mpi_sim.create_ProdUpdate(make_key(op.B), make_key(op.Y))
                self.add_dot_inc(make_key(op.A), make_key(op.X), make_key(op.Y))

            elif op_type == builder.SimFilterSynapse:
                logger.debug(
                    "Creating Filter, input:%d, output:%d, numer:%s, denom:%s",
                    make_key(op.input), make_key(op.output), str(op.num), str(op.den))

                self.mpi_sim.create_Filter(
                    make_key(op.input), make_key(op.output), op.num, op.den)

            elif op_type == builder.SimNeurons:
                n_neurons = op.J.size

                if type(op.neurons) is LIF:
                    tau_ref = op.neurons.tau_ref
                    tau_rc = op.neurons.tau_rc

                    logger.debug(
                        "Creating LIF, N: %d, J:%d, output:%d",
                        n_neurons, make_key(op.J), make_key(op.output))

                    self.mpi_sim.create_SimLIF(
                        n_neurons, tau_rc, tau_ref, self.dt,
                        make_key(op.J), make_key(op.output))

                elif type(op.neurons) is LIFRate:
                    tau_ref = op.neurons.tau_ref
                    tau_rc = op.neurons.tau_rc

                    logger.debug(
                        "Creating LIFRate, N: %d, J:%d, output:%d",
                        n_neurons, make_key(op.J), make_key(op.output))

                    self.mpi_sim.create_SimLIFRate(
                        n_neurons, tau_rc, tau_ref, self.dt,
                        make_key(op.J), make_key(op.output))
                else:
                    raise NotImplementedError(
                        'nengo_mpi cannot handle neurons of type ' +
                        str(type(op.neurons)))

            elif op_type == builder.SimPyFunc:
                t_in = op.t_in
                fn = op.fn
                x = op.x

                output_id = make_key(op.output) if op.output is not None else -1

                if x is None:
                    logger.debug("Creating PyFunc, output:%d", make_key(op.output))

                    if op.output is None:
                        self.mpi_sim.create_PyFunc(fn, t_in)
                    else:
                        self.mpi_sim.create_PyFuncO(
                            output_id, make_checked_func(fn, t_in, False), t_in)

                else:
                    logger.debug(
                        "Creating PyFuncWithInput, output:%d", make_key(op.output))

                    if op.output is None:
                        self.mpi_sim.create_PyFuncI(fn, t_in, make_key(x), x.value)
                    else:
                        self.mpi_sim.create_PyFuncIO(
                            output_id, make_checked_func(fn, t_in, True),
                            t_in, make_key(x), x.value)

            else:
                raise NotImplementedError(
                    'nengo_mpi cannot handle operator of type ' + str(op_type))

            if hasattr(op, 'tag'):
                logger.debug("tag: %s", op.tag)

        self._probe_outputs = self.model.params

        for probe in self.model.probes:
            self.add_probe(
                probe, make_key(self.model.sig[probe]['in']),
                sample_every=probe.sample_every)

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
