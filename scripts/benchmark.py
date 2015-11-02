import logging
import argparse

import numpy as np

import nengo
import nengo_mpi
from nengo_mpi.partition import metis_partitioner, work_balanced_partitioner
from nengo_mpi.partition import random_partitioner, EnsembleArraySplitter

logger = logging.getLogger(__name__)
nengo.log(debug=False)

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
    '--pfunc', type=str, default='',
    help='Specify the algorithm to use for partitioning. '
         'Possible values are: default, metis, random, work.'
         'If not supplied, an assignment scheme is used.')

parser.add_argument(
    '--save', nargs='?', type=str, default='', const='bench',
    help="Supply a filename to write the network to (so it can be "
         "later be used by the stand-alone version of nengo_mpi). "
         "In this case, the network will not be simulated.")

parser.add_argument(
    '--mpi-log', nargs='?', type=str,
    default='', const='bench', dest='mpi_log',
    help="Supply a filename to write the results of the simulation "
         "to, if an MPI simulation is performed.")

parser.add_argument(
    '--ea', action='store_true',
    help="Supply to use ensemble arrays instead of ensembles. Each "
         "ensemble within each ensemble array will have dimension 1, "
         "and npd neurons.")

parser.add_argument(
    '--split-ea', type=int, default=1,
    help="Number of parts to split each ensemble array up into. "
         "Obviously, only has an effect if --ea is also supplied.")

parser.add_argument(
    '--seed', type=int, default=None,
    help="Seed for random number generation.")


args = parser.parse_args()
print "Parameters are: ", args

name = 'MpiBenchmarkNetwork'
N = args.npd
D = args.d
seed = args.seed

n_streams = args.ns
stream_length = args.sl

n_processors = args.p
use_ea = args.ea
split_ea = args.split_ea

use_mpi = args.mpi

save_file = args.save
if save_file == 'bench':
    save_file = (
        'bench_p{0}_sl{1}_ns{2}.net'.format(
            args.p, stream_length, n_streams))

mpi_log = args.mpi_log
if mpi_log == 'bench':
    mpi_log = (
        'bench_p{0}_sl{1}_ns{2}.h5'.format(
            args.p, stream_length, n_streams))

if mpi_log:
    print "Logging simulation results to", mpi_log

sim_time = args.t

progress_bar = not args.noprog

partitioner = args.pfunc

if not partitioner:
    partitioner = None
    denom = int(np.ceil(float(stream_length) / n_processors))
else:
    fmap = {
        'default': None,
        'metis': metis_partitioner, 'random': random_partitioner,
        'work': work_balanced_partitioner}

    partitioner = nengo_mpi.Partitioner(n_processors, func=fmap[partitioner])

assert n_streams > 0
assert stream_length > 0
assert n_processors >= 1
assert sim_time > 0

assignments = {}

ensembles = []

m = nengo.Network(label=name, seed=seed)
with m:
    m.config[nengo.Ensemble].neuron_type = nengo.LIF()
    input_node = nengo.Node(output=[0.25] * D)
    input_p = nengo.Probe(input_node, synapse=0.01)

    probes = []
    for i in range(n_streams):
        ensembles.append([])

        for j in range(stream_length):
            if use_ea:
                ensemble = nengo.networks.EnsembleArray(
                    N, D, label="stream %d, index %d" % (i, j))
                if j > 0:
                    nengo.Connection(ensembles[-1][-1].output, ensemble.input)
                else:
                    nengo.Connection(input_node, ensemble.input)
            else:
                ensemble = nengo.Ensemble(
                    N * D, dimensions=D, label="stream %d, index %d" % (i, j))
                if j > 0:
                    nengo.Connection(ensembles[-1][-1], ensemble)
                else:
                    nengo.Connection(input_node, ensemble)

            if n_processors > 1 and partitioner is None:
                assignments[ensemble] = j / denom

            ensembles[-1].append(ensemble)

        if use_ea:
            ensembles[-1][-1].add_output(
                'zero', function=lambda x: 0)
            nengo.Connection(
                ensembles[-1][-1].zero, ensembles[-1][0].input)
            probes.append(
                nengo.Probe(ensemble.output, synapse=0.01))
        else:
            nengo.Connection(
                ensembles[-1][-1], ensembles[-1][0],
                function=lambda x: np.zeros(D))
            probes.append(
                nengo.Probe(ensemble, synapse=0.01))

if use_ea and split_ea > 1:
    splitter = EnsembleArraySplitter()
    max_neurons = np.ceil(float(D) / split_ea) * N
    splitter.split(m, max_neurons)

if use_mpi:
    if partitioner is not None:
        sim = nengo_mpi.Simulator(
            m, dt=0.001, partitioner=partitioner, save_file=save_file)
    else:
        sim = nengo_mpi.Simulator(
            m, dt=0.001, assignments=assignments, save_file=save_file)

    if save_file:
        print "Saved network to", save_file
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

    n_neurons = N * D * n_streams * stream_length
    print "Total simulation time:", t1 - t0, "seconds"
    print "Parameters were: ", args
    print "Number of neurons in simulations: ", n_neurons

    import pandas as pd
    import os

    try:
        runtimes_file = "/scratch/c/celiasmi/e2crawfo/benchmark_runtimes.csv"
        header = not os.path.isfile(runtimes_file)

        vals = vars(args).copy()
        vals['runtime'] = t1 - t0
        vals['n_neurons'] = n_neurons

        now = pd.datetime.now()
        df = pd.DataFrame(vals, index=pd.date_range(now, periods=1))

        with open(runtimes_file, 'a') as f:
            df.to_csv(f, header=header)
    except:
        print "Could not write runtimes files."
