import logging

import nengo
import numpy as np

import argparse

logger = logging.getLogger(__name__)
nengo.log(debug=False)

parser = argparse.ArgumentParser(
    description="Benchmarking script for nengo_mpi")

parser.add_argument(
    '--ns', type=int, default=1,
    help='Number of streams in the network.')

parser.add_argument(
    '--sl', type=int, default=1,
    help='Length of each stream.')

parser.add_argument(
    '--d', type=int, default=1,
    help='Number of dimensions in each neural ensemble')

parser.add_argument(
    '--npd', type=int, default=50,
    help='Number of neurons per dimension in each neural ensemble')

parser.add_argument(
    '--t', type=float, default=1.0,
    help='Length of the simulation in seconds')

parser.add_argument(
    '--mpi', type=int, default=1, help='Whether to use MPI')

parser.add_argument(
    '--p', type=int, default=1,
    help='If using MPI, the number of processors to use '
         '(components in the partition.')

parser.add_argument(
    '--noprog', action='store_true', default=False,
    help='Supply to omit the progress bar')

args = parser.parse_args()

name = 'node_to_ensemble'
N = args.npd
D = args.d
seed = 10

num_streams = args.ns
stream_length = args.sl

extra_partitions = args.p - 1

use_mpi = args.mpi

sim_time = args.t

progress_bar = not args.noprog

assert num_streams > 0
assert stream_length > 0
assert extra_partitions >= 0
assert sim_time > 0

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

#for key in assignments.keys():
    #assignments[key] = (
    #    0 if assignments[key] == 0
    #    else np.random.choice(partitions) + 1)

if use_mpi:
    import nengo_mpi
    partitioner = nengo_mpi.Partitioner(1 + extra_partitions, assignments)

    sim = nengo_mpi.Simulator(m, dt=0.001, partitioner=partitioner)
else:
    sim = nengo.Simulator(m, dt=0.001)

import time

t0 = time.time()
sim.run(sim_time, progress_bar)
t1 = time.time()

print "Total simulation time:", t1 - t0, "seconds"
print "Parameters were: ", args
print "Number of neurons in simulations: ", N * D * num_streams * stream_length

if probe:
    print "Input node result: "
    print sim.data[input_p][-5:]

    for i, p in enumerate(probes):
        print "Stream %d result: " % i
        print sim.data[p][-5:]
