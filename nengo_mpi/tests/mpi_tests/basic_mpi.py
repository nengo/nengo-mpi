import nengo
import nengo_mpi

import numpy as np

n_neurons = 40
synapse = 0.05
neuron_type = nengo.LIF

m = nengo.Network(seed=1)
with m:
    A = nengo.Ensemble(
        n_neurons, dimensions=3, neuron_type=neuron_type())

    B = nengo.Ensemble(
        n_neurons, dimensions=3, neuron_type=neuron_type())

    nengo.Connection(A, B, synapse=synapse)

    A_p = nengo.Probe(A)
    B_p = nengo.Probe(B)

    input = nengo.Node([0.1, 0.2, -0.3])
    nengo.Connection(input, A, synapse=0.05)

sim_time = 1

refimpl_sim = nengo.Simulator(m)
refimpl_sim.run(sim_time)

sim = nengo_mpi.Simulator(m, partitioner=nengo_mpi.Partitioner(2))
sim.run(sim_time)

assert np.allclose(
    refimpl_sim.data[A_p], sim.data[A_p],
    atol=0.00001, rtol=0.00)
assert np.allclose(
    refimpl_sim.data[B_p], sim.data[B_p],
    atol=0.00001, rtol=0.00)
