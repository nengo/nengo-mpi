import logging

import numpy as np

import nengo
import nengo.utils.numpy as npext
from nengo.dists import Choice
import nengo_mpi

logger = logging.getLogger(__name__)

N = 80
dt2 = 0.002
f = lambda t: np.sin(2 * np.pi * t)
seed = 1

m = nengo.Network(seed=seed)
with m:
    m.config[nengo.Ensemble].neuron_type = nengo.neurons.LIFRate()
    sin = nengo.Node(output=f)
    cons = nengo.Node(output=-.5)
    factors = nengo.Ensemble(
        2 * N, dimensions=2, radius=1.5,
        encoders=Choice([[1, 1], [-1, 1], [1, -1], [-1, -1]]))
    product = nengo.Ensemble(N, dimensions=1)
    nengo.Connection(sin, factors[0])
    nengo.Connection(cons, factors[1])
    nengo.Connection(
        factors, product, function=lambda x: x[0] * x[1], synapse=0.01)

    input1_p = nengo.Probe(sin, 'output', sample_every=dt2, synapse=0.01)
    input2_p = nengo.Probe(cons, 'output', sample_every=dt2, synapse=0.01)
    factors_p = nengo.Probe(factors, 'decoded_output', sample_every=dt2, synapse=0.01)
    product_p = nengo.Probe(product, 'decoded_output', sample_every=dt2, synapse=0.01)

sim = nengo_mpi.Simulator(m)
sim.run(.1)
t = sim.trange(dt=dt2)

print sim.data[input1_p][-10:]
print sim.data[input2_p][-10:]
print sim.data[factors_p][-10:]
print sim.data[product_p][-10:]

assert npext.rmse(sim.data[factors_p][:, 0], f(t)) < 0.1
assert npext.rmse(sim.data[factors_p][20:, 1], -0.5) < 0.1
assert npext.rmse(sim.data[product_p][:, 0], -0.5 * f(t)) < 0.1
