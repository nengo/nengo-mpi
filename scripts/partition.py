
import logging

import numpy as np

import nengo

logger = logging.getLogger(__name__)

dims = 3
n_neurons = 60
radius = 1.0
seed = 1

rng = np.random
rng.seed(seed)

a = rng.uniform(low=-0.7, high=0.7, size=dims)
b = rng.uniform(low=-0.7, high=0.7, size=dims)
c = np.zeros(2 * dims)
c[::2] = a
c[1::2] = b

print "Building network..."
model = nengo.Network(seed=seed)
with model:
    inputA = nengo.Node(a)
    inputB = nengo.Node(b)
    A = nengo.networks.EnsembleArray(n_neurons, dims, radius=radius, label="A")
    B = nengo.networks.EnsembleArray(n_neurons, dims, radius=radius, label="B")
    C = nengo.networks.EnsembleArray(
        n_neurons * 2, dims, ens_dimensions=2, radius=radius, label="C")
    D = nengo.networks.EnsembleArray(
        n_neurons, dims, radius=radius, label="D")

    nengo.Connection(inputA, A.input)
    nengo.Connection(inputB, B.input)
    nengo.Connection(A.output, C.input[::2])
    nengo.Connection(B.output, C.input[1::2])
    nengo.Connection(C.output[:dims], D.input)
    nengo.Connection(D.output, A.input)

    A_p = nengo.Probe(A.output, synapse=0.03)
    B_p = nengo.Probe(B.output, synapse=0.03)
    C_p = nengo.Probe(C.output, synapse=0.03)

from nengo_mpi.partition import metis_partitioner

print "Partitioning network..."
assignments = metis_partitioner(model, 3)

print assignments

print "Done partitioning."