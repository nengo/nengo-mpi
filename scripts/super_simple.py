import logging

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--usempi", action="store_true")
args = parser.parse_args()

print "Using mpi", args.usempi

import numpy as np

import nengo
import nengo.utils.numpy as npext
from nengo.dists import Choice

logger = logging.getLogger(__name__)

N = 10
dt2 = 0.002
f = lambda t: np.sin(2 * np.pi * t)
seed = 1

m = nengo.Network(seed=seed)
with m:
    m.config[nengo.Ensemble].neuron_type = nengo.neurons.LIFRate()
    sin = nengo.Node(output=f)
    product = nengo.Ensemble(N, dimensions=1)
    nengo.Connection(sin, product, synapse=None)

    input1_p = nengo.Probe(sin, 'output', synapse=None)
    product_p = nengo.Probe(product, 'decoded_output', synapse=None)
    ens_input = nengo.Probe(product, 'input', synapse=None)
    neuron_input = nengo.Probe(product.neurons, 'input', synapse=None)

if args.usempi:
    import nengo_mpi
    sim = nengo_mpi.Simulator(m)
else:
    sim = nengo.Simulator(m)

sim.run(.1)
t = sim.trange(dt=dt2)

print sim.data[input1_p][0:10]
print sim.data[ens_input][0:10]
print sim.data[product_p][0:10]
print sim.data[neuron_input][0:10]

print sim.data[product].encoders
