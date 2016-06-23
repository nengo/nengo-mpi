"""
Test that setting cross_at_updates=False has the desired effect.
Namely, that ensembles are allowed to be on different processors even
if they are connected by a Connection that doesn't have a synapse,
and that the results are correct when this happens.

"""

import nengo
import nengo_mpi
from nengo_mpi.partition import work_balanced_partitioner

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
    nengo.Connection(B, A, synapse=None)

    A_p = nengo.Probe(A)
    B_p = nengo.Probe(B)

    input = nengo.Node([0.1, 0.2, -0.3])
    nengo.Connection(input, A, synapse=None)

sim_time = 1

refimpl_sim = nengo.Simulator(m)
refimpl_sim.run(sim_time)

partitioner = nengo_mpi.Partitioner(
    2, cross_at_updates=False, func=work_balanced_partitioner)
sim = nengo_mpi.Simulator(m, partitioner=partitioner)

active_components = set(partitioner.object_assignments.values())
assert 0 in active_components and 1 in active_components

sim.run(sim_time)

assert np.allclose(
    refimpl_sim.data[A_p], sim.data[A_p],
    atol=0.00001, rtol=0.00)
assert np.allclose(
    refimpl_sim.data[B_p], sim.data[B_p],
    atol=0.00001, rtol=0.00)
