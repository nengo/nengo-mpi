import logging

import numpy as np

import nengo
from nengo_mpi import SpaunStimulus

logger = logging.getLogger(__name__)

dims = 4
n_neurons = 25
seed = 1
dimension = 10
radius = 1.0

np.random.seed(seed)

network = nengo.Network(label="SpaunStimulusNetwork", seed=seed)
network.config[nengo.Ensemble].neuron_type = nengo.LIFRate()

with network:
    spaun_stimulus = SpaunStimulus(dimension, ["0", "1", "2"])
    A = nengo.networks.EnsembleArray(n_neurons, dimension, radius=radius)

    nengo.Connection(spaun_stimulus, A.input)

    ss_p = nengo.Probe(spaun_stimulus)
    A_p = nengo.Probe(A.output)

import nengo_mpi
sim = nengo_mpi.Simulator(network)

sim.run(10)

print "SpaunStimulus"
print sim.data[ss_p]

print "A"
print sim.data[A_p]