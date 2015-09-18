import nengo_mpi
import nengo

import numpy as np

import pytest

from nengo.neurons import LIF, LIFRate, RectifiedLinear, Sigmoid
from nengo.neurons import AdaptiveLIF, AdaptiveLIFRate  # , Izhikevich

all_neurons = [
    LIF, LIFRate, RectifiedLinear, Sigmoid,
    AdaptiveLIF, AdaptiveLIFRate]  # Izhikevich]


@pytest.mark.parametrize("neuron_type", all_neurons)
@pytest.mark.parametrize("synapse", [None, 0.0, 0.02, 0.05])
def test_exact_match(Simulator, neuron_type, synapse):
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

    mpi_sim = Simulator(m)
    mpi_sim.run(sim_time)

    assert np.allclose(
        refimpl_sim.data[A_p], mpi_sim.data[A_p], atol=0.00001, rtol=0.00)
    assert np.allclose(
        refimpl_sim.data[B_p], mpi_sim.data[B_p], atol=0.00001, rtol=0.00)


def test_against_refimpl(Simulator):
    """
    Test against the reference implementation in a simple case.
    Require that they be very close to one another.
    """
    seed = 1

    network = nengo.Network(seed=seed)

    with network:
        node = nengo.Node(0.5)
        ens = nengo.Ensemble(100, 1)

        nengo.Connection(node, ens, synapse=0.01)
        probe = nengo.Probe(ens, synapse=0.01)

    mpi_sim = Simulator(network)
    sim = nengo.Simulator(network)

    sim_time = 1.0

    mpi_sim.run(sim_time)

    sim.run(sim_time)

    assert np.allclose(mpi_sim.data[probe][-10:], 0.5, atol=0.4, rtol=0.0)
    assert np.allclose(
        mpi_sim.data[probe][-10:], sim.data[probe][-10:],
        atol=0.00001, rtol=0.0)


def test_close_basic():
    network = nengo.Network()

    with network:
        node = nengo.Node(0.5)
        ens = nengo.Ensemble(100, 1)

        nengo.Connection(node, ens, synapse=0.01)
        nengo.Probe(ens, synapse=0.01)

    assert nengo_mpi.Simulator.all_closed()

    mpi_sim = nengo_mpi.Simulator(network)
    mpi_sim.run(0.1)

    assert not nengo_mpi.Simulator.all_closed()
    mpi_sim.close()
    assert nengo_mpi.Simulator.all_closed()

    mpi_sim = nengo_mpi.Simulator(network)
    mpi_sim.run(0.1)

    assert not nengo_mpi.Simulator.all_closed()
    mpi_sim.close()


def test_too_many_open():
    network = nengo.Network()

    with network:
        node = nengo.Node(0.5)
        ens = nengo.Ensemble(100, 1)

        nengo.Connection(node, ens, synapse=0.01)
        nengo.Probe(ens, synapse=0.01)

    mpi_sim = nengo_mpi.Simulator(network)
    mpi_sim.run(0.1)
    with pytest.raises(RuntimeError) as e:
        mpi_sim = nengo_mpi.Simulator(network)

    assert str(e.value).startswith("Attempting to create")

    mpi_sim.close()
    assert nengo_mpi.Simulator.all_closed()


def test_context_manager():
    network = nengo.Network()

    with network:
        node = nengo.Node(0.5)
        ens = nengo.Ensemble(100, 1)

        nengo.Connection(node, ens, synapse=0.01)
        nengo.Probe(ens, synapse=0.01)

    assert nengo_mpi.Simulator.all_closed()

    with nengo_mpi.Simulator(network) as mpi_sim:
        assert not nengo_mpi.Simulator.all_closed()
        mpi_sim.run(0.1)

    assert nengo_mpi.Simulator.all_closed()

    mpi_sim = nengo_mpi.Simulator(network)
    mpi_sim.run(0.1)
    assert not nengo_mpi.Simulator.all_closed()
    mpi_sim.close()

    assert nengo_mpi.Simulator.all_closed()

    with nengo_mpi.Simulator(network) as mpi_sim:
        assert not nengo_mpi.Simulator.all_closed()
        mpi_sim.run(0.1)

    assert nengo_mpi.Simulator.all_closed()
