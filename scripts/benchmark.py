import logging

import nengo
import numpy as np

logger = logging.getLogger(__name__)
nengo.log(debug=False)

name = 'node_to_ensemble'
N = 50
D = 4
seed = 10

num_streams = 30
stream_length = 30
extra_partitions = 15
partitions = range(extra_partitions)

assignments = {}

probe = True

ensembles = []

m = nengo.Network(label=name, seed=seed)
with m:
    m.config[nengo.Ensemble].neuron_type = nengo.LIF()
    input_node = nengo.Node(output=[0.25] * D)
    input_p = nengo.Probe(input_node, synapse=0.01)

    probes = []
    for i in range(num_streams):
        ensembles.append([])

        for j in range(stream_length):
            ensemble = nengo.Ensemble(
                N * D, dimensions=D, label="stream %d, index %d" % (i, j))

            if j > 0:
                nengo.Connection(ensembles[-1][-1], ensemble)
            else:
                nengo.Connection(input_node, ensemble)

            if extra_partitions:
                assignments[ensemble] = (
                    j / int(np.ceil(float(stream_length) / (extra_partitions + 1))))

            ensembles[-1].append(ensemble)

        nengo.Connection(
            ensembles[-1][-1], ensembles[-1][0],
            function=lambda x: np.zeros(D))

        if probe:
            probes.append(
                nengo.Probe(ensemble, 'decoded_output', synapse=0.01))


assignment_seed = 11

np.random.seed(assignment_seed)
choice = np.random.choice(range(num_streams * stream_length))

#for key in assignments.keys():
#    assignments[key] = (
#        1 if assignments[key] == choice
#        else 0)

    #assignments[key] = (
    #    0 if assignments[key] == 0
    #    else np.random.choice(partitions) + 1)

print assignments.values()

sim_time = 1

if 1:
    import nengo_mpi
    partitioner = nengo_mpi.Partitioner(1 + extra_partitions, assignments)

    sim = nengo_mpi.Simulator(m, dt=0.001, partitioner=partitioner)
else:
    sim = nengo.Simulator(m, dt=0.001)

import time

t0 = time.time()
sim.run(sim_time)
t1 = time.time()

print "Total simulation time:", t1 - t0, "seconds"

if probe:
    print "Input node result: "
    print sim.data[input_p][-5:]

    for i, p in enumerate(probes):
        print "Stream %d result: " % i
        print sim.data[p][-5:]
