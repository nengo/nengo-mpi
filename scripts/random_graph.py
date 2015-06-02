import logging

import nengo
import numpy as np
import nengo_mpi
from nengo_mpi.partition import metis_partitioner, work_balanced_partitioner
from nengo_mpi.partition import spectral_partitioner, random_partitioner

import argparse

logger = logging.getLogger(__name__)
nengo.log(debug=False)

parser = argparse.ArgumentParser(
    description="Benchmarking script for nengo_mpi. "
                "Network takes the form of a randomly generated graph, where "
                "each node is a nengo ensemble.")

parser.add_argument(
    '-n', type=int, default=10,
    help='Number of ensembles in the network.')

parser.add_argument(
    '-q', type=float, default=0.1,
    help='Probability of each directed edge in the graph existing.')

parser.add_argument(
    '-d', type=int, default=1,
    help='Number of dimensions in each neural ensemble.')

parser.add_argument(
    '--npd', type=int, default=50,
    help='Number of neurons per dimension in each neural ensemble.')

parser.add_argument(
    '--mpi', type=int, default=1, help='Whether to use MPI.')

parser.add_argument(
    '-p', type=int, default=1,
    help='If using MPI, the number of processors to use.')

parser.add_argument(
    '--probes', type=float, default=0.1,
    help='Percentage of ensembles that are probed.')

parser.add_argument(
    '--self', type=float, default=0.1,
    help='Percentage of ensembles with self-loops.')

parser.add_argument(
    '-t', type=float, default=1.0,
    help='Length of the simulation in seconds.')

parser.add_argument(
    '--noprog', action='store_true',
    help='Supply to omit the progress bar.')

parser.add_argument(
    '--pfunc', type=str, default='',
    help='Specify the algorithm to use for partitioning. '
         'Possible values are: default, metis, random, spectral, work.')

parser.add_argument(
    '--save', nargs='?', type=str, default='', const='random_graph',
    help="Supply a filename to write the network to (so it can be "
         "later be used by the stand-alone version of nengo_mpi). "
         "In this case, the network will not be simulated.")

parser.add_argument(
    '--mpi-log', nargs='?', type=str,
    default='', const='random_graph', dest='mpi_log',
    help="Supply a filename to write the results of the simulation "
         "to, if an MPI simulation is performed.")

args = parser.parse_args()
print "Parameters are: ", args

name = 'MpiRandomGraphBenchmark'
N = args.npd
D = args.d
seed = 10

n_nodes = args.n
pct_connections = args.q
n_processors = args.p
pct_probed = args.probes
pct_self_loops = args.self

use_mpi = args.mpi

save_file = args.save
if save_file == 'random_graph':
    save_file = (
        'random_graph_p{0}_n{1}_q{2}.net'.format(
            args.p, n_nodes, pct_connections))

mpi_log = args.mpi_log
if mpi_log == 'random_graph':
    mpi_log = (
        'random_graph_p{0}_n{1}_q{2}.h5'.format(
            args.p, n_nodes, pct_connections))

if mpi_log:
    print "Logging simulation results to", mpi_log

sim_time = args.t

progress_bar = not args.noprog

partitioner = args.pfunc

assert n_nodes > 0
assert pct_connections >= 0 and pct_connections <= 1
assert pct_self_loops >= 0 and pct_self_loops <= 1
assert pct_probed >= 0 and pct_probed <= 1
assert n_processors >= 1
assert sim_time > 0

ensembles = []
n_directed_connections = 0

rng = np.random.RandomState(seed)

m = nengo.Network(label=name, seed=seed)
with m:
    m.config[nengo.Ensemble].neuron_type = nengo.LIF()
    input_node = nengo.Node(output=[0.25] * D)
    input_p = nengo.Probe(input_node, synapse=0.01)

    probes = []

    for i in range(n_nodes):
        ensemble = nengo.Ensemble(
            N * D, dimensions=D, label="ensemble %d" % i)

        if rng.rand() < pct_self_loops:
            nengo.Connection(ensemble, ensemble)

        if rng.rand() < pct_probed:
            probe = nengo.Probe(ensemble)
            probes.append(probe)

        ensembles.append(ensemble)

        for j in range(i):
            if rng.rand() < pct_connections:
                nengo.Connection(ensemble, ensembles[j])
                n_directed_connections += 1

            if rng.rand() < pct_connections:
                nengo.Connection(ensembles[j], ensemble)
                n_directed_connections += 1

    nengo.Connection(input_node, ensembles[0])
    n_directed_connections += 1

if use_mpi:
    fmap = {
        'default': None, '': None,
        'metis': metis_partitioner, 'spectral': spectral_partitioner,
        'random': random_partitioner, 'work': work_balanced_partitioner}

    partitioner = nengo_mpi.Partitioner(n_processors, func=fmap[partitioner])

    sim = nengo_mpi.Simulator(
        m, dt=0.001, partitioner=partitioner, save_file=save_file)

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
            print "Result from %s: " % probe
            print sim.data[p][-10:]

    n_neurons = N * D * n_nodes
    print "Total simulation time:", t1 - t0, "seconds"
    print "Parameters were: ", args
    print "Number of neurons in network: ", n_neurons
    print "Number of nodes in network: ", n_nodes
    print (
        "Number of directed connections in network: "
        "%s" % n_directed_connections)

    import pandas as pd
    import os

    try:
        runtimes_file = (
            "/scratch/c/celiasmi/e2crawfo/random_graph_runtimes.csv")
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
