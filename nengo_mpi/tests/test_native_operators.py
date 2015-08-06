import nengo_mpi
from nengo_mpi.model import MpiModel
from nengo_mpi.spaun_mpi import SpaunStimulusOperator

import nengo
from nengo.builder.signal import Signal
from nengo.builder.operator import DotInc, ElementwiseInc, Reset, Copy
from nengo.builder.node import SimPyFunc
from nengo.builder.neurons import SimNeurons
from nengo.builder.synapses import SimSynapse

from nengo.neurons import LIF, LIFRate, RectifiedLinear, Sigmoid
from nengo.neurons import AdaptiveLIF, AdaptiveLIFRate  # , Izhikevich

from nengo.synapses import Lowpass  # , LinearFilter, Alpha

from nengo.simulator import ProbeDict
import nengo.utils.numpy as npext

import numpy as np
from collections import defaultdict

import pytest


def Mpi2Simulator(*args, **kwargs):
    return TestSimulator(*args, **kwargs)


def pytest_funcarg__Simulator(request):
    """The Simulator class being tested."""
    return Mpi2Simulator


class SignalProbe(object):
    def __init__(self, signal, sample_every=None):
        self._signal = signal
        self.sample_every = sample_every

    @property
    def signal(self):
        return self._signal


class TestSimulator(nengo_mpi.Simulator):
    """
    Simulator for testing C++ operators.

    Parameters
    ----------

    operators: list
        List of python operators.

    signal_probes: list
        List of SignalProbes.
    """

    def __init__(
            self, operators, signal_probes, dt=0.001, seed=None):

        # Note: seed is not used right now, but one day...
        assert seed is None, "Simulator seed not yet implemented"
        self.seed = np.random.randint(npext.maxint) if seed is None else seed

        self.n_steps = 0
        self.dt = dt

        assignments = defaultdict(int)

        print "Building MPI model..."
        self.model = MpiModel(
            1, assignments, dt=dt, label="TestSimulator", free_memory=False)

        self.model.add_ops(0, operators)

        for probe in signal_probes:
            self.model.sig[probe]['in'] = probe.signal

        self.model.probes = signal_probes

        print "Finalizing MPI model..."
        self.model.finalize_build()

        # probe -> python list
        self._probe_outputs = self.model.params

        self.data = ProbeDict(self._probe_outputs)

        print "TestSimulator ready."

    def run_steps(self, steps, progress_bar=False, log_filename=""):
        """Simulate for the given number of `dt` steps."""

        if progress_bar:
            raise Exception(
                "TestSimulator does not allow showing the progress bar.")

        if log_filename:
            raise Exception("TestSimulator cannot use log-files.")

        super(TestSimulator, self).run_steps(steps, False, "")

    def run(self, time_in_seconds, progress_bar=False, log_filename=""):
        """Simulate for the given length of time."""

        if progress_bar:
            raise Exception(
                "TestSimulator does not allow showing the progress bar.")

        if log_filename:
            raise Exception("TestSimulator cannot use log-files.")

        super(TestSimulator, self).run(time_in_seconds, False, "")


def test_dot_inc(Simulator):
    seed = 1
    np.random.seed(seed)

    D = 2

    A = Signal(np.eye(D), 'A')
    X = Signal(np.ones(D), 'X')
    Y = Signal(np.zeros(D), 'Y')

    ops = [Reset(Y), DotInc(A, X, Y)]

    signal_probes = [SignalProbe(Y)]

    sim = Simulator(ops, signal_probes)

    sim.run(1)

    assert np.allclose(A.value.dot(X.value), sim.data[signal_probes[0]])


def test_random_dot_inc(Simulator):
    seed = 1
    np.random.seed(seed)

    D = 3
    n_tests = 10

    for i in range(n_tests):
        A = Signal(np.random.random((D, D)), 'A')
        X = Signal(np.random.random(D), 'X')
        Y = Signal(np.zeros(D), 'Y')

        ops = [Reset(Y), DotInc(A, X, Y)]

        probes = [SignalProbe(Y)]

        sim = Simulator(ops, probes)

        sim.run(1)

        assert np.allclose(A.value.dot(X.value), sim.data[probes[0]])


def test_reset(Simulator):

    D = 40
    reset_val = 4.0

    A = Signal(np.ones((D, D)), 'A')
    X = Signal(np.ones(D), 'X')
    Y = Signal(np.zeros(D), 'Y')

    ops = [Reset(Y, reset_val), DotInc(A, X, Y)]
    probes = [SignalProbe(Y)]
    sim = Simulator(ops, probes)

    sim.run(0.05)

    assert np.allclose(D + reset_val, sim.data[probes[0]])


def test_copy(Simulator):
    D = 40
    copy_val = 4.0

    A = Signal(np.ones((D, D)), 'A')
    X = Signal(np.ones(D), 'X')
    Y = Signal(np.zeros(D), 'Y')

    C = Signal(copy_val * np.ones(D))

    ops = [Copy(Y, C), DotInc(A, X, Y)]
    probes = [SignalProbe(Y)]
    sim = Simulator(ops, probes)

    sim.run(0.05)

    assert np.allclose(D + copy_val, sim.data[probes[0]])


def test_lif(Simulator):
    """Test that the dynamic lif model approximately matches the rates."""

    n_neurons = 40
    tau_rc = 0.02
    tau_ref = 0.002
    lif = LIF(tau_rc=tau_rc, tau_ref=tau_ref)

    voltage = Signal(
        np.zeros(n_neurons), name="%s.voltage" % lif)

    ref_time = Signal(
        np.zeros(n_neurons), name="%s.refractory_time" % lif)

    J = Signal(np.zeros(n_neurons), 'J')
    output = Signal(np.zeros(n_neurons), 'output')

    op = SimNeurons(
        neurons=lif, J=J, output=output, states=[voltage, ref_time])

    input = np.arange(-2, 2, .1)
    input_func = SimPyFunc(J, lambda: input, False, None)

    probes = [SignalProbe(output)]

    sim = Simulator([op, input_func], probes)
    sim.run(1.0)

    spikes = sim.data[probes[0]] / (1.0 / sim.dt)
    sim_rates = spikes.sum(axis=0)

    math_rates = lif.rates(
        input, gain=np.ones(n_neurons), bias=np.zeros(n_neurons))

    assert np.allclose(np.squeeze(sim_rates), math_rates, atol=1, rtol=0.02)


@pytest.mark.parametrize("neuron_type", [LIFRate, Sigmoid, RectifiedLinear])
def test_stateless_neurons(Simulator, neuron_type):
    """
    Test that the mpi version of a neuron
    model matches the ref-impl version.
    """
    neurons = neuron_type()

    n_neurons = 40

    J = Signal(np.zeros(n_neurons), 'J')
    output = Signal(np.zeros(n_neurons), 'output')

    op = SimNeurons(neurons=neurons, J=J, output=output)

    input = np.arange(-2, 2, .1)
    input_func = SimPyFunc(J, lambda: input, False, None)

    probes = [SignalProbe(output)]

    sim = Simulator([op, input_func], probes)
    sim.run(0.2)

    sim_rates = sim.data[probes[0]]
    math_rates = neurons.rates(
        input, gain=np.ones(n_neurons), bias=np.zeros(n_neurons))

    assert np.allclose(
        np.squeeze(sim_rates[-1, :]), math_rates, atol=0.0001, rtol=0.02)

all_neurons = [
    LIF, LIFRate, RectifiedLinear, Sigmoid,
    AdaptiveLIF, AdaptiveLIFRate]  # Izhikevich]


@pytest.mark.parametrize("neuron_type", all_neurons)
@pytest.mark.parametrize("synapse", [None, 0.0, 0.02, 0.05])
def test_exact_match(neuron_type, synapse):
    n_neurons = 40

    sequence = np.random.random((1000, 3))

    def f(t):
        val = sequence[int(t * 1000)]
        return val

    m = nengo.Network(seed=1)
    with m:
        A = nengo.Ensemble(
            n_neurons, dimensions=3, neuron_type=neuron_type())

        B = nengo.Ensemble(
            n_neurons, dimensions=3, neuron_type=neuron_type())

        nengo.Connection(A, B, synapse=synapse)

        A_p = nengo.Probe(A)
        B_p = nengo.Probe(B)

        input = nengo.Node(f)
        nengo.Connection(input, A, synapse=0.05)

    sim_time = 0.01

    refimpl_sim = nengo.Simulator(m)
    refimpl_sim.run(sim_time)

    mpi_sim = nengo_mpi.Simulator(m)
    mpi_sim.run(sim_time)

    assert np.allclose(
        refimpl_sim.data[A_p], mpi_sim.data[A_p], atol=0.00001, rtol=0.00)
    assert np.allclose(
        refimpl_sim.data[B_p], mpi_sim.data[B_p], atol=0.00001, rtol=0.00)


def test_filter(Simulator):
    D = 1

    input = Signal(np.zeros(D), 'input')
    output = Signal(np.zeros(D), 'output')

    synapse = SimSynapse(input, output, Lowpass(0.05))
    input_func = SimPyFunc(input, lambda: 1, False, None)

    probes = [SignalProbe(output)]

    sim = Simulator([synapse, input_func], probes)
    sim.run(0.2)


def test_element_wise_inc(Simulator):
    M = 3
    N = 2

    X = Signal(3 * np.ones(1), 'X')
    A = Signal(2 * np.ones((M, 1)), 'A')
    Y = Signal(np.zeros((M, 1)), 'Y')
    ewi_helper(Simulator, A, X, Y, 6)

    X = Signal(3 * np.ones((M, 1)), 'X')
    A = Signal(2 * np.ones(1), 'A')
    Y = Signal(np.zeros((M, 1)), 'Y')
    ewi_helper(Simulator, A, X, Y, 6)

    X = Signal(3 * np.ones(1), 'X')
    A = Signal(2 * np.ones((1, N)), 'A')
    Y = Signal(np.zeros((M, 1)), 'Y')
    ewi_helper(Simulator, A, X, Y, 6)

    X = Signal(3 * np.ones((1, N)), 'X')
    A = Signal(2 * np.ones(1), 'A')
    Y = Signal(np.zeros((M, 1)), 'Y')
    ewi_helper(Simulator, A, X, Y, 6)

    X = Signal(3 * np.ones((M, 1)), 'X')
    A = Signal(2 * np.ones((M, 1)), 'A')
    Y = Signal(np.zeros((M, 1)), 'Y')
    ewi_helper(Simulator, A, X, Y, 6)

    X = Signal(3 * np.ones((1, N)), 'X')
    A = Signal(2 * np.ones((1, N)), 'A')
    Y = Signal(np.zeros((1, N)), 'Y')
    ewi_helper(Simulator, A, X, Y, 6)

    X = Signal(3 * np.ones(1), 'X')
    A = Signal(2 * np.ones((M, N)), 'A')
    Y = Signal(np.zeros((M, N)), 'Y')
    ewi_helper(Simulator, A, X, Y, 6)

    X = Signal(3 * np.ones((M, N)), 'X')
    A = Signal(2 * np.ones(1), 'A')
    Y = Signal(np.zeros((M, N)), 'Y')
    ewi_helper(Simulator, A, X, Y, 6)

    X = Signal(3 * np.ones((M, N)), 'X')
    A = Signal(2 * np.ones((M, N)), 'A')
    Y = Signal(np.zeros((M, N)), 'Y')
    ewi_helper(Simulator, A, X, Y, 6)

    X = Signal(3 * np.ones((M, 1)), 'X')
    A = Signal(2 * np.ones((1, N)), 'A')
    Y = Signal(np.zeros((M, N)), 'Y')
    ewi_helper(Simulator, A, X, Y, 6)

    X = Signal(3 * np.ones((M, 1)), 'X')
    A = Signal(2 * np.ones((M, N)), 'A')
    Y = Signal(np.zeros((M, N)), 'Y')
    ewi_helper(Simulator, A, X, Y, 6)

    X = Signal(3 * np.ones((M, N)), 'X')
    A = Signal(2 * np.ones((M, 1)), 'A')
    Y = Signal(np.zeros((M, N)), 'Y')
    ewi_helper(Simulator, A, X, Y, 6)

    X = Signal(3 * np.ones((1, N)), 'X')
    A = Signal(2 * np.ones((M, N)), 'A')
    Y = Signal(np.zeros((M, N)), 'Y')
    ewi_helper(Simulator, A, X, Y, 6)

    X = Signal(3 * np.ones((M, N)), 'X')
    A = Signal(2 * np.ones((1, N)), 'A')
    Y = Signal(np.zeros((M, N)), 'Y')
    ewi_helper(Simulator, A, X, Y, 6)

    X = Signal(3 * np.ones((1, 1)), 'X')
    A = Signal(2 * np.ones((1, 1)), 'A')
    Y = Signal(np.zeros((1, 1)), 'Y')
    ewi_helper(Simulator, A, X, Y, 6)


def ewi_helper(Simulator, A, X, Y, reference):
    ewi = ElementwiseInc(A, X, Y)

    probes = [SignalProbe(Y)]

    sim = Simulator([ewi, Reset(Y)], probes)
    sim.run(0.2)
    assert np.allclose(reference, sim.data[probes[0]][-1])


def test_spaun_stimulus(Simulator):
    spaun_vision = pytest.importorskip("_spaun.vision.lif_vision")
    spaun_config = pytest.importorskip("_spaun.config")
    spaun_stimulus = pytest.importorskip("_spaun.modules.stimulus")
    cfg = spaun_config.cfg

    cfg.raw_seq_str = 'A0[#1]?X'
    spaun_stimulus.parse_raw_seq()

    dimension = spaun_vision.images_data_dimensions
    output = Signal(np.zeros(dimension), name="spaun stimulus output")

    spaun_stim = SpaunStimulusOperator(
        output, cfg.raw_seq, cfg.present_interval, cfg.present_blanks)

    probes = [SignalProbe(output)]
    operators = [spaun_stim]

    sim = Simulator(operators, probes)
    run_time = spaun_stimulus.get_est_runtime()

    sim.run(run_time)

    def image_string(image):
        s = int(np.sqrt(len(image)))
        string = ""
        for i in range(s):
            for j in range(s):
                string += str(1 if image[i * s + j] > 0 else 0)
            string += '\n'
        return string

    p = probes[0]

    model = nengo.Network()
    with model:
        output = nengo.Node(
            output=spaun_stimulus.stim_func_vis, label='Stim Module Out')
        ref_p = nengo.Probe(output)

    ref_sim = nengo.Simulator(model)
    ref_sim.run(spaun_stimulus.get_est_runtime())

    interval = int(1000 * cfg.present_interval)

    assert ref_sim.data[ref_p].shape[0] == sim.data[p].shape[0]

    compare_indices = range(0, 2) + range(interval-2, interval)
    compare_indices += range(2*interval-2, 2*interval)
    compare_indices += range(3*interval-2, 3*interval-2)
    compare_indices += range(4*interval-1, 4*interval)
    compare_indices += range(5*interval-2, 5*interval)

    for i in compare_indices:
        if not all(ref_sim.data[ref_p][i] == sim.data[p][i]):
            print "Mismatch between stimulus at index: ", i
            print image_string(ref_sim.data[ref_p][i])
            print image_string(sim.data[p][i])
            raise Exception()

if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
