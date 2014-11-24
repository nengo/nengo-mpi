import logging
import numpy as np
import pytest

import nengo_mpi
import mpi_sim

import nengo
from nengo.simulator import ProbeDict
from nengo.utils.numpy import rmse
import nengo.utils.numpy as npext

logger = logging.getLogger(__name__)


class TestSimulator(nengo_mpi.Simulator):
    """Simulator for testing individual ops."""

    def __init__(self, init_func, dt=0.001, seed=None):
        """
        init_func accepts a simulator_chunk, and should add the ops it wants
        to test to that chunk by directly calling the simulator_chunk member
        functions.
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

        self.mpi_sim = mpi_sim.PythonMpiSimulator()

        mpi_chunk = self.mpi_sim.add_chunk()

        simulator_chunk = nengo_mpi.chunk.SimulatorChunk(
            mpi_chunk, dt=dt, probe_keys=self.probe_keys,
            probe_outputs=self._probe_outputs)

        init_func(simulator_chunk)

        self.mpi_sim.finalize()

        self.data = ProbeDict(self._probe_outputs)


def Mpi2Simulator(*args, **kwargs):
    return TestSimulator(*args, **kwargs)


def pytest_funcarg__Simulator(request):
    """The Simulator class being tested."""
    return Mpi2Simulator


def test_reset(Simulator, plt):

    D = 40
    reset_val = 4.0

    def init_func(sim_chunk):
        A = np.random.random((D, D))
        X = 2 * np.ones(D)
        Y = np.zeros(D)

        sim_chunk.add_signal(0, A)
        sim_chunk.add_signal(1, X)
        sim_chunk.add_signal(2, Y)

        sim_chunk.mpi_chunk.create_DotInc(0, 1, 2)
        sim_chunk.mpi_chunk.create_Reset(2, reset_val)
        sim_chunk.add_probe(2, 2)

    sim = Simulator(init_func)
    sim.run(1.0)

    t = sim.trange(dt=0.001)
    plt.plot(t, sim.data[2])
    plt.savefig('mpi.test_operators.test_reset.pdf')
    plt.close()

    assert rmse(sim.data[2], reset_val) < 0.001


def test_copy(Simulator, plt):

    D = 40

    all_data = []

    def make_random():
        data = np.random.random(D)
        all_data.append(data)
        return data

    def init_func(sim_chunk):

        A = np.random.random((D, D))
        X = 2 * np.ones(D)
        Y = np.zeros(D)
        Z = np.random.random(D)

        sim_chunk.add_signal(0, A)
        sim_chunk.add_signal(1, X)
        sim_chunk.add_signal(2, Y)
        sim_chunk.add_signal(3, Z)

        sim_chunk.mpi_chunk.create_PyFuncO(3, make_random, False)
        sim_chunk.mpi_chunk.create_DotInc(0, 1, 2)
        sim_chunk.mpi_chunk.create_Copy(2, 3)

        sim_chunk.add_probe(2, 2)
        sim_chunk.add_probe(3, 3)

    sim = Simulator(init_func)
    sim.run(1.0)

    t = sim.trange(dt=.001)
    plt.plot(t, sim.data[2][:, 0])
    plt.plot(t, sim.data[3][:, 0])
    plt.legend(loc='best')
    plt.savefig('mpi.test_operators.test_copy.pdf')
    plt.close()

    all_data = np.array(all_data)
    assert rmse(sim.data[2], sim.data[3]) < 0.001
    assert rmse(np.squeeze(sim.data[2]), all_data) < 0.001


def test_lif(Simulator, plt):
    """Test that the dynamic lif model approximately matches the rates."""
    n_neurons = 40
    tau_rc = 0.02
    tau_ref = 0.002
    dt = 0.001

    J = np.arange(-2, 2, .1)

    def make_random():
        return J

    def init_func(sim_chunk):

        A = np.zeros(n_neurons)
        B = np.zeros(n_neurons)

        sim_chunk.add_signal(0, A)
        sim_chunk.add_signal(1, B)

        sim_chunk.mpi_chunk.create_PyFuncO(0, make_random, False)
        sim_chunk.mpi_chunk.create_SimLIF(n_neurons, tau_rc, tau_ref, dt, 0, 1)
        sim_chunk.add_probe(1, 1)

    sim = Simulator(init_func)
    sim.run(1.0)

    spikes = sim.data[1]
    sim_rates = spikes.sum(0)

    lif = nengo.LIF(tau_rc=tau_rc, tau_ref=tau_ref)
    math_rates = lif.rates(
        J, gain=np.ones(n_neurons), bias=np.zeros(n_neurons))

    plt.plot(J, sim_rates, label='sim')
    plt.plot(J, math_rates, label='math')

    plt.legend(loc='best')
    plt.savefig('mpi.test_operators.test_lif.pdf')
    plt.close()

    assert np.allclose(np.squeeze(sim_rates), math_rates, atol=2, rtol=0.02)


def test_lif_rate(Simulator, plt):
    """Test that the dynamic model approximately matches the rates."""
    n_neurons = 40
    tau_rc = 0.02
    tau_ref = 0.002
    dt = 0.001

    J = np.arange(-2, 2, .1)

    def make_random():
        return J

    def init_func(sim_chunk):
        A = np.zeros(n_neurons)
        B = np.zeros(n_neurons)

        sim_chunk.add_signal(0, A)
        sim_chunk.add_signal(1, B)

        sim_chunk.mpi_chunk.create_PyFuncO(0, make_random, False)
        sim_chunk.mpi_chunk.create_SimLIFRate(
            n_neurons, tau_rc, tau_ref, dt, 0, 1)
        sim_chunk.add_probe(1, 1)

    sim = Simulator(init_func)
    sim.run(1.0)

    output = sim.data[1]
    sim_rates = output[-1, :] / dt

    lif = nengo.LIF(tau_rc=tau_rc, tau_ref=tau_ref)
    math_rates = lif.rates(
        J, gain=np.ones(n_neurons), bias=np.zeros(n_neurons))

    plt.plot(J, sim_rates, label='sim')
    plt.plot(J, math_rates, label='math')
    plt.legend(loc='best')
    plt.savefig('mpi.test_operators.test_lif_rate.pdf')
    plt.close()

    assert np.allclose(np.squeeze(sim_rates), math_rates, atol=1, rtol=0.02)


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
