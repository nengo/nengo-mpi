import logging

import nengo
import numpy as np

import argparse

logger = logging.getLogger(__name__)
nengo.log(debug=True)

parser = argparse.ArgumentParser(
    description="Benchmarking script for nengo_mpi.")

parser.add_argument(
    '--ns', type=int, default=1,
    help='Number of streams in the network.')

parser.add_argument(
    '--sl', type=int, default=1,
    help='Length of each stream.')

parser.add_argument(
    '-d', type=int, default=1,
    help='Number of dimensions in each neural ensemble.')

parser.add_argument(
    '--npd', type=int, default=50,
    help='Number of neurons per dimension in each neural ensemble.')

parser.add_argument(
    '-t', type=float, default=1.0,
    help='Length of the simulation in seconds.')

parser.add_argument(
    '--mpi', type=int, default=1, help='Whether to use MPI.')

parser.add_argument(
    '-p', type=int, default=1,
    help='If using MPI, the number of processors to use.')

parser.add_argument(
    '--noprog', action='store_true',
    help='Supply to omit the progress bar.')

parser.add_argument(
    '--rand', action='store_true',
    help='Supply to use a (pseudo) random scheme for assigning nengo '
         'object to processors')

parser.add_argument(
    '--save', type=str, default='',
    help="Supply a filename to write the network to (so it can be "
         "later be used by the stand-alone version of nengo_mpi). "
         "In this case, the network will not be simulated.")

parser.add_argument(
    '--mpi-log', type=str, default='', dest='mpi_log',
    help="Supply a filename to write the results of the simulation "
         "to, if an MPI simulation is performed.")

args = parser.parse_args()
print "Parameters are: ", args

name = 'MpiBenchmarkNetwork'
N = args.npd
D = args.d
seed = 10

num_streams = args.ns
stream_length = args.sl

extra_partitions = args.p - 1

use_mpi = args.mpi
mpi_log = args.mpi_log

save_file = args.save

sim_time = args.t

progress_bar = not args.noprog

random_partitions = args.rand

assert num_streams > 0
assert stream_length > 0
assert extra_partitions >= 0
assert sim_time > 0

assignment_seed = 11
assignments_rng = np.random.RandomState(assignment_seed)
denom = int(np.ceil(float(stream_length) / (extra_partitions + 1)))

assignments = {}

ensembles = []

m = nengo.Network(label=name, seed=seed)
with m:
    m.config[nengo.Ensemble].neuron_type = nengo.LIFRate()
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
                if random_partitions:
                    assignments[ensemble] = (
                        assignments_rng.randint(extra_partitions + 1))
                else:
                    assignments[ensemble] = j / denom

            ensembles[-1].append(ensemble)

        nengo.Connection(
            ensembles[-1][-1], ensembles[-1][0],
            function=lambda x: np.zeros(D))

        probes.append(
            nengo.Probe(ensemble, 'decoded_output', synapse=0.01))

if use_mpi:
    import nengo_mpi
    partitioner = nengo_mpi.Partitioner(1 + extra_partitions, assignments)

    sim = nengo_mpi.Simulator(
        m, dt=0.001, partitioner=partitioner, save_file=save_file)
else:
    sim = nengo.Simulator(m, dt=0.001)

if not save_file:
    import time

    t0 = time.time()

    if use_mpi:
        sim.run(sim_time, progress_bar, mpi_log)
    else:
        sim.run(sim_time, progress_bar)

    t1 = time.time()

    if not mpi_log:
        print "Input node result: "
        print sim.data[input_p][-10:]

        for i, p in enumerate(probes):
            print "Stream %d result: " % i
            print sim.data[p][-10:]

    num_neurons = N * D * num_streams * stream_length
    print "Total simulation time:", t1 - t0, "seconds"
    print "Parameters were: ", args
    print "Number of neurons in simulations: ", num_neurons

    import pandas as pd
    import os

    try:
        runtimes_file = "/scratch/c/celiasmi/e2crawfo/benchmark_runtimes.csv"
        header = not os.path.isfile(runtimes_file)

        vals = vars(args).copy()
        vals['runtime'] = t1 - t0
        vals['num_neurons'] = num_neurons

        now = pd.datetime.now()
        df = pd.DataFrame(vals, index=pd.date_range(now, periods=1))

        with open(runtimes_file, 'a') as f:
            df.to_csv(f, header=header)
    except:
        print "Could not write runtimes files."
