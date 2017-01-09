import nengo_mpi

import nengo
from nengo.neurons import LIF, LIFRate, RectifiedLinear, Sigmoid
from nengo.neurons import AdaptiveLIF, AdaptiveLIFRate  # , Izhikevich
from nengo.tests.test_learning_rules import learning_net
from nengo.learning_rules import Voja

import numpy as np
import pytest

all_neurons = [
    LIF, LIFRate, RectifiedLinear, Sigmoid,
    AdaptiveLIF, AdaptiveLIFRate]  # Izhikevich]


def test_doc_example(Simulator):
    with nengo.Network(seed=1) as m:
        sin_input = nengo.Node(output=0.5)

        sin_ens = nengo.Ensemble(n_neurons=100, dimensions=1)
        nengo.Connection(sin_input, sin_ens)

        sin_squared = nengo.Ensemble(n_neurons=100, dimensions=1)
        nengo.Connection(sin_ens, sin_squared, function=np.square)

        squared_probe = nengo.Probe(sin_squared, synapse=0.01)

    sim_time = 1.0

    refimpl_sim = nengo.Simulator(m)
    refimpl_sim.run(sim_time)

    mpi_sim = Simulator(m)
    mpi_sim.run(sim_time)

    mpi_squared = mpi_sim.data[squared_probe]
    ref_squared = refimpl_sim.data[squared_probe]

    assert np.allclose(ref_squared, mpi_squared, atol=0.00001, rtol=0.00)


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


def test_seeding():
    network = nengo.Network()

    with network:
        proc1 = nengo.processes.WhiteNoise()
        node1 = nengo.Node(proc1)
        probe1 = nengo.Probe(node1)

        proc2 = nengo.processes.WhiteNoise()
        node2 = nengo.Node(proc2)
        probe2 = nengo.Probe(node2)

        proc3 = nengo.processes.WhiteNoise()
        node3 = nengo.Node(proc3)
        probe3 = nengo.Probe(node3)

    seed = 10

    try:
        sim = nengo_mpi.Simulator(network, seed=seed)
        sim.run(0.1)

        d1 = sim.data[probe1]
        d2 = sim.data[probe2]
        d3 = sim.data[probe3]

        # Make sure that processes are getting different seeds
        assert not np.allclose(d1, d2, atol=0.00001, rtol=0.00)
        assert not np.allclose(d2, d3, atol=0.00001, rtol=0.00)

        sim.reset(seed)
        sim.run(0.1)

        # Make sure processes give same results after reset
        assert np.allclose(d1, sim.data[probe1], atol=0.00001, rtol=0.00)
        assert np.allclose(d2, sim.data[probe2], atol=0.00001, rtol=0.00)
        assert np.allclose(d3, sim.data[probe3], atol=0.00001, rtol=0.00)

        sim.reset(seed+1)
        sim.run(0.1)

        # Make sure changing the seed changes the results
        assert not np.allclose(d1, sim.data[probe1], atol=0.00001, rtol=0.00)
        assert not np.allclose(d2, sim.data[probe2], atol=0.00001, rtol=0.00)
        assert not np.allclose(d3, sim.data[probe3], atol=0.00001, rtol=0.00)

    finally:
        try:
            sim.close()
        except:
            pass


def test_spaun_stim():
    spaun_vision = pytest.importorskip("_spaun.vision.lif_vision")
    spaun_config = pytest.importorskip("_spaun.config")
    spaun_stimulus = pytest.importorskip("_spaun.modules.stimulus")
    spaun_modules = pytest.importorskip("_spaun.modules")

    cfg = spaun_config.cfg

    cfg.raw_seq_str = '#1#2'
    spaun_stimulus.parse_raw_seq()

    dimension = spaun_vision.images_data_dimensions

    network = nengo.Network()
    with network:
        stim1 = nengo_mpi.spaun_mpi.SpaunStimulus(
            dimension, cfg.raw_seq, cfg.present_interval,
            cfg.present_blanks)
        stim2 = nengo_mpi.spaun_mpi.SpaunStimulus(
            dimension, cfg.raw_seq, cfg.present_interval,
            cfg.present_blanks)

        assert stim1.identifier != stim2.identifier, (
            "Identifier assignment failed.")

        n1 = nengo.Node(size_in=dimension)
        n2 = nengo.Node(size_in=dimension)
        n3 = nengo.Node(size_in=dimension)
        n4 = nengo.Node(size_in=dimension)

        nengo.Connection(stim1, n1)
        nengo.Connection(stim1, n2)

        nengo.Connection(stim2, n3)
        nengo.Connection(stim2, n4)

        p1 = nengo.Probe(n1)
        p2 = nengo.Probe(n2)
        p3 = nengo.Probe(n3)
        p4 = nengo.Probe(n4)

    seed = 10
    runtime = spaun_modules.get_est_runtime()

    try:
        sim = nengo_mpi.Simulator(network, seed=seed)
        sim.run(runtime)

        d1 = sim.data[p1]
        d2 = sim.data[p2]
        d3 = sim.data[p3]
        d4 = sim.data[p4]

        assert np.allclose(d1, d2, atol=0.00001, rtol=0.00)
        assert np.allclose(d3, d4, atol=0.00001, rtol=0.00)

        assert not np.allclose(d1, d3, atol=0.00001, rtol=0.00)

        sim.reset(seed)
        sim.run(runtime)

        assert np.allclose(d1, sim.data[p1], atol=0.00001, rtol=0.00)
        assert np.allclose(d2, sim.data[p2], atol=0.00001, rtol=0.00)
        assert np.allclose(d3, sim.data[p3], atol=0.00001, rtol=0.00)
        assert np.allclose(d4, sim.data[p4], atol=0.00001, rtol=0.00)

        sim.reset(seed+1)
        sim.run(runtime)

        assert not np.allclose(d1, sim.data[p1], atol=0.00001, rtol=0.00)
        assert not np.allclose(d2, sim.data[p2], atol=0.00001, rtol=0.00)
        assert not np.allclose(d3, sim.data[p3], atol=0.00001, rtol=0.00)
        assert not np.allclose(d4, sim.data[p4], atol=0.00001, rtol=0.00)
    finally:
        try:
            sim.close()
        except:
            pass


@pytest.mark.parametrize("learning_rule", [nengo.BCM, nengo.Oja])
def test_unsupervised_exact_match(Simulator, learning_rule, seed, rng):
    sim_time = 1.0

    m, activity_p, trans_p = learning_net(
        learning_rule, nengo.Network(seed=seed), rng)

    refimpl_sim = nengo.Simulator(m)
    refimpl_sim.run(sim_time)

    mpi_sim = Simulator(m)
    mpi_sim.run(sim_time)

    assert np.allclose(
        refimpl_sim.data[activity_p], mpi_sim.data[activity_p],
        atol=0.00001, rtol=0.00)
    assert np.allclose(
        refimpl_sim.data[trans_p], mpi_sim.data[trans_p],
        atol=0.00001, rtol=0.00)

    refimpl_sim.reset()
    mpi_sim.reset()

    refimpl_sim.run(sim_time)
    mpi_sim.run(sim_time)

    assert np.allclose(
        refimpl_sim.data[activity_p], mpi_sim.data[activity_p],
        atol=0.00001, rtol=0.00)
    assert np.allclose(
        refimpl_sim.data[trans_p], mpi_sim.data[trans_p],
        atol=0.00001, rtol=0.00)


def test_voja_exact_match(Simulator, nl_nodirect, seed, rng):
    n = 200
    learned_vector = np.asarray([0.5])

    def control_signal(t):
        """Modulates the learning on/off."""
        return 0 if t < 0.5 else -1

    m = nengo.Network(seed=seed)
    with m:
        m.config[nengo.Ensemble].neuron_type = nl_nodirect()
        control = nengo.Node(output=control_signal)
        u = nengo.Node(output=learned_vector)
        x = nengo.Ensemble(n, dimensions=len(learned_vector))

        conn = nengo.Connection(
            u, x, synapse=None, learning_rule_type=Voja(None))
        nengo.Connection(control, conn.learning_rule, synapse=None)

        p_enc = nengo.Probe(conn.learning_rule, 'scaled_encoders')

    sim_time = 1.0

    refimpl_sim = nengo.Simulator(m)
    refimpl_sim.run(sim_time/2)
    refimpl_sim.run(sim_time/2)

    mpi_sim = Simulator(m)
    mpi_sim.run(sim_time/2)
    mpi_sim.run(sim_time/2)

    assert np.allclose(
        refimpl_sim.data[p_enc], mpi_sim.data[p_enc],
        atol=0.00001, rtol=0.00)

    refimpl_sim.reset()
    mpi_sim.reset()

    refimpl_sim.run(sim_time)
    mpi_sim.run(sim_time)

    assert np.allclose(
        refimpl_sim.data[p_enc], mpi_sim.data[p_enc],
        atol=0.00001, rtol=0.00)
