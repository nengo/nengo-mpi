import nengo_mpi
from nengo_mpi.model import MpiModel
from nengo_mpi.spaun_mpi import SpaunStimulusOperator

import nengo
from nengo.builder.signal import Signal
from nengo.builder.operator import DotInc, ElementwiseInc, Reset
from nengo.builder.operator import Copy, SlicedCopy, TimeUpdate
from nengo.builder.node import SimPyFunc
from nengo.builder.neurons import SimNeurons
from nengo.builder.processes import SimProcess

from nengo.neurons import LIF, LIFRate, RectifiedLinear, Sigmoid

from nengo.synapses import Lowpass  # , LinearFilter, Alpha

from nengo.simulator import ProbeDict
import nengo.utils.numpy as npext

import numpy as np
from collections import defaultdict
import logging
logger = logging.getLogger(__name__)

import pytest


class SignalProbe(object):
    def __init__(self, signal, sample_every=None):
        self._signal = signal
        self.sample_every = sample_every

    @property
    def signal(self):
        return self._signal


class TestSimulator(nengo_mpi.Simulator):
    """ Simulator for testing C++ operators.

    Note that this Simulator does not need to be closed as
    nengo_mpi.Simulator does.

    Parameters
    ----------
    operators: list
        List of python operators.
    signal_probes: list
        List of SignalProbes.

    """
    def __init__(self, operators, signal_probes, dt=0.001, seed=None):

        if self._open_simulators:
            raise RuntimeError(
                "Attempting to create active instance of TestSimulator "
                "while another instance exists that has not been "
                "closed. Call `close` on existing instances before "
                "creating new ones.")

        self.runnable = True
        self.n_steps = 0
        self.dt = dt

        assignments = defaultdict(int)

        print "Building MPI model..."
        self.model = MpiModel(1, assignments, dt=dt, label="TestSimulator")

        self.model.assign_ops(0, operators)

        for probe in signal_probes:
            self.model.sig[probe]['in'] = probe.signal

        self.model.probes = signal_probes

        print "Finalizing MPI model..."
        self.model.finalize_build()

        # probe -> python list
        self._probe_outputs = self.model.params

        self.data = ProbeDict(self._probe_outputs)

        self._open_simulators.append(self)
        seed = np.random.randint(npext.maxint) if seed is None else seed
        self.reset(seed=seed)

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


def test_dot_inc():
    seed = 1
    np.random.seed(seed)

    D = 2

    A = Signal(np.random.random((D, D)), 'A')
    X = Signal(np.random.random(D), 'X')
    Y = Signal(np.zeros(D), 'Y')

    ops = [Reset(Y), DotInc(A, X, Y)]
    probes = [SignalProbe(Y)]

    with TestSimulator(ops, probes) as sim:
        sim.run(0.01)

    assert np.allclose(
        A.initial_value.dot(X.initial_value), sim.data[probes[0]])


def test_random_dot_inc():
    seed = 1
    np.random.seed(seed)

    D = 10
    n_tests = 10

    for i in range(n_tests):
        A = Signal(np.random.random((D, D)), 'A')
        X = Signal(np.random.random(D), 'X')
        Y = Signal(np.zeros(D), 'Y')

        ops = [Reset(Y), DotInc(A, X, Y)]
        probes = [SignalProbe(Y)]

        with TestSimulator(ops, probes) as sim:
            sim.run(0.01)

        assert np.allclose(
            A.initial_value.dot(X.initial_value), sim.data[probes[0]])


def test_reset():

    D = 40
    reset_val = 4.0

    A = Signal(np.ones((D, D)), 'A')
    X = Signal(np.ones(D), 'X')
    Y = Signal(np.zeros(D), 'Y')

    ops = [Reset(Y, reset_val), DotInc(A, X, Y)]
    probes = [SignalProbe(Y)]
    with TestSimulator(ops, probes) as sim:
        sim.run(0.05)

    assert np.allclose(
        D + reset_val, sim.data[probes[0]], atol=0.00001, rtol=0.0)


def test_copy():
    D = 40
    copy_val = 4.0

    A = Signal(np.ones((D, D)), 'A')
    X = Signal(np.ones(D), 'X')
    Y = Signal(np.zeros(D), 'Y')

    C = Signal(copy_val * np.ones(D))

    ops = [Copy(Y, C), DotInc(A, X, Y)]
    probes = [SignalProbe(Y)]
    with TestSimulator(ops, probes) as sim:
        sim.run(0.05)

    assert np.allclose(
        D + copy_val, sim.data[probes[0]], atol=0.00001, rtol=0.0)


def test_sliced_copy_converge():
    D = 40

    data1 = np.random.random(D)
    S1 = Signal(data1, 'S1')

    data2 = np.random.random(D)
    S2 = Signal(data2, 'S2')

    data3 = np.random.random(D)
    S3 = Signal(data3, 'S3')

    converge = Signal(np.zeros(3 * D), 'converge')

    ops = [
        Reset(converge, 0),
        SlicedCopy(S1, converge, b_slice=slice(0, D), inc=True),
        SlicedCopy(S2, converge, b_slice=slice(D, 2 * D), inc=True),
        SlicedCopy(S3, converge, b_slice=slice(2 * D, 3 * D), inc=True)]

    probes = [SignalProbe(converge)]
    with TestSimulator(ops, probes) as sim:
        sim.run(0.05)

    ground_truth = np.array(list(data1) + list(data2) + list(data3))

    assert np.allclose(
        ground_truth, sim.data[probes[0]], atol=0.000001, rtol=0.0)


def test_sliced_copy_diverge():
    D = 40

    data = np.random.random(3 * D)
    diverge = Signal(data, 'diverge')

    S1 = Signal(np.zeros(D), 'S1')
    S2 = Signal(np.zeros(D), 'S2')
    S3 = Signal(np.zeros(D), 'S3')

    ops = [
        Reset(S1, 0),
        Reset(S2, 0),
        Reset(S3, 0),
        SlicedCopy(diverge, S1, a_slice=slice(0, D), inc=True),
        SlicedCopy(diverge, S2, a_slice=slice(D, 2 * D), inc=True),
        SlicedCopy(diverge, S3, a_slice=slice(2 * D, 3 * D), inc=True)]

    probes = [SignalProbe(S1), SignalProbe(S2), SignalProbe(S3)]
    with TestSimulator(ops, probes) as sim:
        sim.run(0.05)

    assert np.allclose(
        data[:D], sim.data[probes[0]], atol=0.000001, rtol=0.0)
    assert np.allclose(
        data[D:2*D], sim.data[probes[1]], atol=0.000001, rtol=0.0)
    assert np.allclose(
        data[2*D:], sim.data[probes[2]], atol=0.000001, rtol=0.0)


def test_sliced_copy():
    D = 2 * 20

    data1 = np.random.random(D)
    S1 = Signal(data1, 'S1')

    data2 = np.random.random(D)
    S2 = Signal(data2, 'S2')

    data3 = np.random.random(D)
    S3 = Signal(data3, 'S3')

    converge = Signal(np.zeros(3 * D), 'converge')

    out1 = Signal(np.zeros(1.5 * D), 'out1')
    out2 = Signal(np.zeros(1.5 * D), 'out2')
    final_out = Signal(np.zeros(2 * D), 'final_out')

    ops = [
        Reset(converge, 0),
        SlicedCopy(S1, converge, b_slice=slice(0, D), inc=True),
        SlicedCopy(S2, converge, b_slice=slice(D, 2 * D), inc=True),
        SlicedCopy(S3, converge, b_slice=slice(2 * D, 3 * D), inc=True),
        SlicedCopy(converge, out1, a_slice=slice(0, int(1.5 * D))),
        SlicedCopy(converge, out2, a_slice=slice(int(1.5 * D), 3*D)),
        Reset(final_out, 0),
        SlicedCopy(
            out1, final_out, a_slice=slice(0, D, 1),
            b_slice=slice(0, 2 * D, 2), inc=True),
        SlicedCopy(
            out2, final_out, a_slice=slice(0, D, 1),
            b_slice=slice(1, 2 * D, 2), inc=True)]

    probes = [SignalProbe(final_out), SignalProbe(converge)]
    with TestSimulator(ops, probes) as sim:
        sim.run(0.05)

    ground_truth = np.zeros(2*D)
    ground_truth[0:2*D:2] = data1
    ground_truth[1:2*D:2] = np.hstack((data2[int(0.5*D):], data3[:int(0.5*D)]))

    assert np.allclose(
        ground_truth, sim.data[probes[0]], atol=0.000001, rtol=0.0)


def test_sliced_copy_seq():
    D = 2 * 20

    permutation1 = list(np.random.permutation(3*D))
    permutation2 = list(np.random.permutation(3*D))
    permutation3 = list(np.random.permutation(int(1.5*D)))
    permutation4 = list(np.random.permutation(int(1.5*D)))
    permutation5 = list(np.random.permutation(3*D))

    data1 = np.random.random(D)
    S1 = Signal(data1, 'S1')

    data2 = np.random.random(D)
    S2 = Signal(data2, 'S2')

    data3 = np.random.random(D)
    S3 = Signal(data3, 'S3')

    converge = Signal(np.zeros(3 * D), 'converge')

    out1 = Signal(np.zeros(1.5 * D), 'out1')
    out2 = Signal(np.zeros(1.5 * D), 'out2')
    final_out = Signal(np.zeros(3 * D), 'final_out')

    ops = [
        Reset(converge, 0),
        SlicedCopy(S1, converge, b_slice=permutation1[:D], inc=True),
        SlicedCopy(S2, converge, b_slice=permutation1[D:2*D], inc=True),
        SlicedCopy(S3, converge, b_slice=permutation1[2*D:], inc=True),
        SlicedCopy(
            converge, out1, a_slice=permutation2[:int(1.5*D)], inc=False),
        SlicedCopy(
            converge, out2, a_slice=permutation2[int(1.5*D):], inc=False),
        Reset(final_out, 0),
        SlicedCopy(
            out1, final_out, a_slice=permutation3,
            b_slice=permutation5[:int(1.5*D)], inc=True),
        SlicedCopy(
            out2, final_out, a_slice=permutation4,
            b_slice=permutation5[int(1.5*D):], inc=True)]

    probes = [SignalProbe(final_out), SignalProbe(converge)]
    with TestSimulator(ops, probes) as sim:
        sim.run(0.05)

    data = np.hstack((data1, data2, data3))

    permuted_once = np.zeros(3 * D)
    permuted_once[permutation1] = data

    permuted_twice = np.zeros(3 * D)
    permuted_twice[:] = permuted_once[permutation2]

    permuted_thrice = np.zeros(3 * D)

    permuted_thrice[permutation5[:int(1.5*D)]] = (
        permuted_twice[:int(1.5*D)][permutation3])

    permuted_thrice[permutation5[int(1.5*D):]] = (
        permuted_twice[int(1.5*D):][permutation4])

    assert np.allclose(
        permuted_thrice, sim.data[probes[0]], atol=0.000001, rtol=0.0)


def test_lif():
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
    input_func = SimPyFunc(J, lambda: input, None, None)

    probes = [SignalProbe(output)]

    with TestSimulator([op, input_func], probes) as sim:
        sim.run(1.0)

    spikes = sim.data[probes[0]] / (1.0 / sim.dt)
    sim_rates = spikes.sum(axis=0)

    math_rates = lif.rates(
        input, gain=np.ones(n_neurons), bias=np.zeros(n_neurons))

    assert np.allclose(np.squeeze(sim_rates), math_rates, atol=1, rtol=0.02)


@pytest.mark.parametrize("neuron_type", [LIFRate, Sigmoid, RectifiedLinear])
def test_stateless_neurons(neuron_type):
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
    input_func = SimPyFunc(J, lambda: input, None, None)

    probes = [SignalProbe(output)]

    with TestSimulator([op, input_func], probes) as sim:
        sim.run(0.2)

    sim_rates = sim.data[probes[0]]
    math_rates = neurons.rates(
        input, gain=np.ones(n_neurons), bias=np.zeros(n_neurons))

    assert np.allclose(
        np.squeeze(sim_rates[-1, :]), math_rates, atol=0.0001, rtol=0.02)


def test_filter():
    D = 1

    input = Signal(np.zeros(D), 'input')
    output = Signal(np.zeros(D), 'output')
    step = Signal(np.array(0, dtype=np.int64), name='step')
    time = Signal(np.array(0, dtype=np.float64), name='time')

    synapse = SimProcess(Lowpass(0.05), input, output, t=time)
    input_func = SimPyFunc(input, lambda: 1, None, None)
    time_update = TimeUpdate(step, time)

    probes = [SignalProbe(output)]

    with TestSimulator(
            [synapse, input_func, time_update], probes) as sim:
        sim.run(0.2)


def test_element_wise_inc():
    M = 3
    N = 2

    A = Signal(2 * np.ones((M, 1)), 'A')
    X = Signal(3 * np.ones(1), 'X')
    Y = Signal(np.zeros((M, 1)), 'Y')
    ewi_helper(TestSimulator, A, X, Y, 6)

    A = Signal(2 * np.ones(1), 'A')
    X = Signal(3 * np.ones((M, 1)), 'X')
    Y = Signal(np.zeros((M, 1)), 'Y')
    ewi_helper(TestSimulator, A, X, Y, 6)

    A = Signal(2 * np.ones((1, N)), 'A')
    X = Signal(3 * np.ones(1), 'X')
    Y = Signal(np.zeros((1, N)), 'Y')
    ewi_helper(TestSimulator, A, X, Y, 6)

    A = Signal(2 * np.ones(1), 'A')
    X = Signal(3 * np.ones((1, N)), 'X')
    Y = Signal(np.zeros((1, N)), 'Y')
    ewi_helper(TestSimulator, A, X, Y, 6)

    A = Signal(2 * np.ones((M, 1)), 'A')
    X = Signal(3 * np.ones((M, 1)), 'X')
    Y = Signal(np.zeros((M, 1)), 'Y')
    ewi_helper(TestSimulator, A, X, Y, 6)

    A = Signal(2 * np.ones((1, N)), 'A')
    X = Signal(3 * np.ones((1, N)), 'X')
    Y = Signal(np.zeros((1, N)), 'Y')
    ewi_helper(TestSimulator, A, X, Y, 6)

    A = Signal(2 * np.ones((M, N)), 'A')
    X = Signal(3 * np.ones(1), 'X')
    Y = Signal(np.zeros((M, N)), 'Y')
    ewi_helper(TestSimulator, A, X, Y, 6)

    A = Signal(2 * np.ones(1), 'A')
    X = Signal(3 * np.ones((M, N)), 'X')
    Y = Signal(np.zeros((M, N)), 'Y')
    ewi_helper(TestSimulator, A, X, Y, 6)

    A = Signal(2 * np.ones((M, N)), 'A')
    X = Signal(3 * np.ones((M, N)), 'X')
    Y = Signal(np.zeros((M, N)), 'Y')
    ewi_helper(TestSimulator, A, X, Y, 6)

    A = Signal(2 * np.ones((1, N)), 'A')
    X = Signal(3 * np.ones((M, 1)), 'X')
    Y = Signal(np.zeros((M, N)), 'Y')
    ewi_helper(TestSimulator, A, X, Y, 6)

    A = Signal(2 * np.ones((M, 1)), 'A')
    X = Signal(3 * np.ones((1, N)), 'X')
    Y = Signal(np.zeros((M, N)), 'Y')
    ewi_helper(TestSimulator, A, X, Y, 6)

    A = Signal(2 * np.ones((M, N)), 'A')
    X = Signal(3 * np.ones((M, 1)), 'X')
    Y = Signal(np.zeros((M, N)), 'Y')
    ewi_helper(TestSimulator, A, X, Y, 6)

    A = Signal(2 * np.ones((M, 1)), 'A')
    X = Signal(3 * np.ones((M, N)), 'X')
    Y = Signal(np.zeros((M, N)), 'Y')
    ewi_helper(TestSimulator, A, X, Y, 6)

    A = Signal(2 * np.ones((M, N)), 'A')
    X = Signal(3 * np.ones((1, N)), 'X')
    Y = Signal(np.zeros((M, N)), 'Y')
    ewi_helper(TestSimulator, A, X, Y, 6)

    A = Signal(2 * np.ones((1, N)), 'A')
    X = Signal(3 * np.ones((M, N)), 'X')
    Y = Signal(np.zeros((M, N)), 'Y')
    ewi_helper(TestSimulator, A, X, Y, 6)

    A = Signal(2 * np.ones((1, 1)), 'A')
    X = Signal(3 * np.ones((1, 1)), 'X')
    Y = Signal(np.zeros((1, 1)), 'Y')
    ewi_helper(TestSimulator, A, X, Y, 6)


def ewi_helper(Simulator, A, X, Y, reference):
    ewi = ElementwiseInc(A, X, Y)

    probes = [SignalProbe(Y)]

    with Simulator([ewi, Reset(Y)], probes) as sim:
        sim.run(0.2)
    assert np.allclose(reference, sim.data[probes[0]][-1])


def test_spaun_stimulus():
    spaun_vision = pytest.importorskip("_spaun.vision.lif_vision")
    spaun_config = pytest.importorskip("_spaun.config")
    spaun_stimulus = pytest.importorskip("_spaun.modules.stimulus")
    cfg = spaun_config.cfg

    cfg.raw_seq_str = 'A0[#1]?X'
    spaun_stimulus.parse_raw_seq()

    dimension = spaun_vision.images_data_dimensions
    output = Signal(np.zeros(dimension), name="spaun stimulus output")

    spaun_stim = SpaunStimulusOperator(
        output, cfg.raw_seq, cfg.present_interval,
        cfg.present_blanks, identifier=0)

    probes = [SignalProbe(output)]
    operators = [spaun_stim]

    run_time = spaun_stimulus.get_est_runtime()
    with TestSimulator(operators, probes) as sim:
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

    # We don't compare when the stimulus is one of the MNIST numbers
    # because these are chosen randomly at runtime, and the random numbers
    # from the reference impl and the mpi impl will not match.
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
