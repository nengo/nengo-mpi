import nengo
import nengo_mpi

import numpy as np

network = nengo.Network(seed=22)

with network:
    node = nengo.Node(0.5)
    A = nengo.Ensemble(100, 1)
    B = nengo.Ensemble(100, 1)
    C = nengo.Ensemble(100, 1)
    D = nengo.Ensemble(100, 1)

    nengo.Connection(node, A, synapse=0.01)
    nengo.Connection(A, B, synapse=0.01)
    nengo.Connection(B, C, synapse=0.01)
    nengo.Connection(C, D, synapse=0.01)
    pA = nengo.Probe(A, synapse=0.01)
    pB = nengo.Probe(B, synapse=0.01)
    pC = nengo.Probe(C, synapse=0.01)
    pD = nengo.Probe(D, synapse=0.01)

assert nengo_mpi.Simulator.all_closed()

sim = nengo.Simulator(network)
mpi_sim = nengo_mpi.Simulator(network, partitioner=nengo_mpi.Partitioner(4))

sim.run(0.1)
mpi_sim.run(0.1)

assert not nengo_mpi.Simulator.all_closed()
mpi_sim.close()

assert np.allclose(
    mpi_sim.data[pA], sim.data[pA], atol=0.00001, rtol=0.0)
assert np.allclose(
    mpi_sim.data[pB], sim.data[pB], atol=0.00001, rtol=0.0)
assert np.allclose(
    mpi_sim.data[pC], sim.data[pC], atol=0.00001, rtol=0.0)
assert np.allclose(
    mpi_sim.data[pD], sim.data[pD], atol=0.00001, rtol=0.0)

assert nengo_mpi.Simulator.all_closed()

sim = nengo.Simulator(network)
mpi_sim = nengo_mpi.Simulator(network, partitioner=nengo_mpi.Partitioner(4))

sim.run(0.1)
mpi_sim.run(0.1)

assert not nengo_mpi.Simulator.all_closed()
mpi_sim.close()
assert nengo_mpi.Simulator.all_closed()

assert np.allclose(
    mpi_sim.data[pA], sim.data[pA], atol=0.00001, rtol=0.0)
assert np.allclose(
    mpi_sim.data[pB], sim.data[pB], atol=0.00001, rtol=0.0)
assert np.allclose(
    mpi_sim.data[pC], sim.data[pC], atol=0.00001, rtol=0.0)
assert np.allclose(
    mpi_sim.data[pD], sim.data[pD], atol=0.00001, rtol=0.0)
