import logging
import argparse
import time
import warnings

import numpy as np

import nengo
from nengo import spa
import nengo_mpi
from nengo_mpi.partition import metis_partitioner, work_balanced_partitioner
from nengo_mpi.partition import random_partitioner

logger = logging.getLogger(__name__)

then_total = time.time()

parser = argparse.ArgumentParser(description="An associative memory.")

parser.add_argument(
    '--n-ensembles', type=int, default=2,
    help='Number of ensembles. If supplied, overrides both sl and ns. '
         'If not a square root, will be round up to the nearest square root.')

parser.add_argument(
    '-d', type=int, default=2,
    help='Number of dimensions of the vectors.')

parser.add_argument(
    '--npe', type=int, default=50,
    help='Number of neurons per neural ensemble.')

parser.add_argument(
    '-t', type=float, default=1.0,
    help='Length of the simulation in seconds.')

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
    '--save', nargs='?', type=str, default='', const='grid',
    help="Supply a filename to write the network to (so it can be "
         "later be used by the stand-alone version of nengo_mpi). "
         "In this case, the network will not be simulated.")

parser.add_argument(
    '--mpi-log', nargs='?', type=str,
    default='', const='grid', dest='mpi_log',
    help="Supply a filename to write the results of the simulation "
         "to, if an MPI simulation is performed.")

parser.add_argument(
    '--seed', type=int, default=None,
    help="Seed for random number generation.")

parser.add_argument(
    '--cross-at-updates', action='store_true',
    help="Force connections that cross component boundaries to have synapses.")


args = parser.parse_args()
print "Parameters are: ", args

name = 'AssociativeMemory'
N = args.npe
D = args.d
seed = args.seed
n_ensembles = args.n_ensembles

n_procs = args.p
use_mpi = n_procs > 0


save_file = args.save
if save_file == 'assoc':
    save_file = (
        'assoc_p{0}_n_ensembles{1}_npe{2}.net'.format(
            args.p, n_ensembles, N))

mpi_log = args.mpi_log
if mpi_log == 'assoc':
    mpi_log = (
        'assoc_p{0}_n_ensembles{1}_npe{2}.h5'.format(
            args.p, n_ensembles, N))

if mpi_log:
    print "Logging simulation results to", mpi_log

sim_time = args.t

progress_bar = not args.noprog

partitioner = args.pfunc

if not partitioner:
    partitioner = None
else:
    fmap = {
        'default': None,
        'metis': metis_partitioner, 'random': random_partitioner,
        'work': work_balanced_partitioner}

    partitioner = nengo_mpi.Partitioner(
        n_procs, func=fmap[partitioner],
        cross_at_updates=args.cross_at_updates)


assert n_ensembles > 0, "Positive number of ensembles required."
assert n_procs >= 0
assert sim_time > 0

n_neurons = N * n_ensembles

rng = np.random.RandomState(seed)

assignments = {}

vocab = spa.Vocabulary(dimensions=D, rng=rng)

words = ['V'+str(i) for i in range(n_ensembles)]
with warnings.catch_warnings():
    warnings.simplefilter('ignore')

    for word in words:
        vocab.parse(word)

with spa.SPA('AssociativeMemory', seed=rng.randint(1000)) as m:
    # create the AM module
    m.assoc_mem = spa.AssociativeMemory(input_vocab=vocab)

    # present input to the AM
    m.am_input = spa.Input(assoc_mem='0')

    # record the inputs and outputs during the simulation
    input_probe = nengo.Probe(m.assoc_mem.input)
    output_probe = nengo.Probe(m.assoc_mem.output, synapse=0.03)

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
    then = time.time()
    sim = nengo.Simulator(m, dt=0.001)
    print "Building network took %f seconds." % (time.time() - then)

if not save_file:
    if use_mpi:
        sim.run(sim_time, progress_bar, mpi_log)
    else:
        then = time.time()
        sim.run(sim_time, progress_bar)
        print "Loading network from file took 0.0 seconds."
        print "Simulating %d steps took %f seconds." % (
            sim.n_steps, (time.time() - then))

    print "Parameters were: ", args
    print "Number of neurons in simulations: ", n_neurons

print (
    "Total time for running script "
    "was %f seconds." % (time.time() - then_total))
