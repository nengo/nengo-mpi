import logging

import numpy as np

import nengo
from nengo.networks.circularconvolution import (
    circconv, transform_in, transform_out)
from nengo.utils.numpy import rmse

logger = logging.getLogger(__name__)

#def test_input_magnitude(Simulator, seed, rng, dims=16, magnitude=10):
#    neurons_per_product = 128
#
#    a = rng.normal(scale=np.sqrt(1./dims), size=dims) * magnitude
#    b = rng.normal(scale=np.sqrt(1./dims), size=dims) * magnitude
#    result = circconv(a, b)
#
#    model = nengo.Network(label="circular conv", seed=seed)
#    model.config[nengo.Ensemble].neuron_type = nengo.LIFRate()
#    with model:
#        inputA = nengo.Node(a)
#        inputB = nengo.Node(b)
#        cconv = nengo.networks.CircularConvolution(
#            neurons_per_product, dimensions=dims,
#            input_magnitude=magnitude)
#        nengo.Connection(inputA, cconv.A, synapse=None)
#        nengo.Connection(inputB, cconv.B, synapse=None)
#        res_p = nengo.Probe(cconv.output)
#        cconv_bad = nengo.networks.CircularConvolution(
#            neurons_per_product, dimensions=dims,
#            input_magnitude=1)  # incorrect magnitude
#        nengo.Connection(inputA, cconv_bad.A, synapse=None)
#        nengo.Connection(inputB, cconv_bad.B, synapse=None)
#        res_p_bad = nengo.Probe(cconv_bad.output)
#    sim = Simulator(model)
#    sim.run(0.01)
#
#    error = rmse(result, sim.data[res_p][-1]) / (magnitude ** 2)
#    error_bad = rmse(result, sim.data[res_p_bad][-1]) / (magnitude ** 2)
#
#    assert error < 0.1
#    assert error_bad > 0.1

dims = 4
neurons_per_product = 32
seed = 1

np.random.seed(seed)

a = np.random.normal(scale=np.sqrt(1./dims), size=dims)
b = np.random.normal(scale=np.sqrt(1./dims), size=dims)
result = circconv(a, b)

model = nengo.Network(label="circular conv", seed=seed)
model.config[nengo.Ensemble].neuron_type = nengo.LIFRate()
with model:
    inputA = nengo.Node(a)
    inputB = nengo.Node(b)

    cconv = nengo.networks.CircularConvolution(
        neurons_per_product, dimensions=dims)
    nengo.Connection(inputA, cconv.A, synapse=None)
    nengo.Connection(inputB, cconv.B, synapse=None)
    res_p = nengo.Probe(cconv.output)

    A_p = nengo.Probe(inputA)
    B_p = nengo.Probe(inputB)
    A_p1 = nengo.Probe(cconv.product.A)
    B_p1 = nengo.Probe(cconv.product.B)

if 1:
    import nengo_mpi
    sim = nengo_mpi.Simulator(model)
else:
    sim = nengo.Simulator(model)

sim.run(0.01)

print sim.data[res_p]
print "A"
print sim.data[A_p]
print "B"
print sim.data[B_p]
print "A1"
print sim.data[A_p1]
print "B1"
print sim.data[B_p1]

print "Real"
print a
print b

error = rmse(result, sim.data[res_p][-1])
assert error < 0.1
