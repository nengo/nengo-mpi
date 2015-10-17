import nengo
from nengo.dists import Gaussian
from nengo.processes import WhiteNoise

import nengo_mpi

import numpy as np

seed = 10
trun = 0.1

with nengo.Network() as model:
    u = nengo.Node(WhiteNoise(Gaussian(0, 1), scale=False))
    up = nengo.Probe(u)

sim = nengo_mpi.Simulator(model, partitioner=nengo_mpi.Partitioner(4))

sim.run(trun)
x = np.array(sim.data[up])

sim.reset()
sim.run(trun)
y = np.array(sim.data[up])

assert x.shape == y.shape
assert (x == y).all()