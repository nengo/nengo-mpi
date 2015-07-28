import nengo_mpi
import nengo

import numpy as np


def test_against_refimpl():
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

    mpi_sim = nengo_mpi.Simulator(network)
    sim = nengo.Simulator(network)

    sim_time = 1.0

    mpi_sim.run(sim_time)
    sim.run(sim_time)

    assert np.allclose(mpi_sim.data[probe][-10:], 0.5, atol=0.4, rtol=0.0)
    assert np.allclose(
        mpi_sim.data[probe][-10:], sim.data[probe][-10:],
        atol=0.00001, rtol=0.0)
