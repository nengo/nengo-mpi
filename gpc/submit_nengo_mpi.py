"""
A script for submitting nengo_mpi experiments to the GPC job queue
using the stand alone nengo mpi executable, nengo_mpi.
Creates a new directory for each experiment in ``experiments_dir'',
which is labelled by date and time the experiment was started.
Launches the script from that directory, and then writes the output
from running the script to ``results.txt'' in that directory.
Also creates a sym link called ``latest'' in ``experiments_dir'',
which points to the directoy of the most recently run experiment.

Example:
"""

import string
import os
import shutil
import subprocess
import argparse
import datetime

parser = argparse.ArgumentParser(
    description="Run nengo mpi simulation using GPC queue.")

parser.add_argument(
    'n', type=int, default=1,
    help="Number of hardware nodes. Same as -N, but doesn't "
         "override arguments to the nengo script.")

parser.add_argument(
    'network', default="", type=str,
    help="The network file to run.")

parser.add_argument(
    't', default=1.0, type=float,
    help="The simulation time.")

parser.add_argument(
    '--show-prog', dest="show_prog", default=0,
    type=int, help="Show the progress bar.")

parser.add_argument(
    '--log', default="", type=str, help="Name of file to store the results.")

parser.add_argument(
    '-w', default="0:15:00",
    help="Upper bound on required wall time.")

args = parser.parse_args()
print args

num_nodes = args.n
num_processors = num_nodes * 8

network_file = args.network
assert network_file, "Must supply a network file to run."

sim_time = args.t

show_prog = args.show_prog

log = args.log

wall_time = args.w

experiments_dir = "/scratch/c/celiasmi/e2crawfo/experiments"
directory = experiments_dir + "/exp_"
date_time_string = str(datetime.datetime.now()).split('.')[0]
date_time_string = reduce(
    lambda y, z: string.replace(y, z, "_"),
    [date_time_string, ":", " ", "-"])
directory += date_time_string
directory += '_p_%d' % (num_nodes * 8)

if not os.path.isdir(directory):
    os.makedirs(directory)

submit_script_name = "submit_script.sh"
results = "results.txt"

worker_location = os.path.join(
    os.getenv("HOME"), 'nengo_mpi/bin/nengo_mpi')


def make_sym_link(target, name):
    try:
        os.remove(name)
    except OSError:
        pass

    os.symlink(target, name)

make_sym_link(directory, experiments_dir+'/latest')

with open(directory + '/' + submit_script_name, 'w') as outf:
    outf.write("#!/bin/bash\n")
    outf.write("# MOAB/Torque submission script for SciNet GPC\n")
    outf.write("#\n")

    line = "#PBS -l nodes=%d:ppn=8,walltime=%s\n" % (num_nodes, wall_time)
    outf.write(line)

    outf.write("#PBS -N nengo_mpi\n")
    outf.write("#PBS -m abe\n\n")

    outf.write(
        "# load modules (must match modules used for compilation)\n")
    outf.write("module load intel/14.0.1\n")
    outf.write("module load python/2.7.5\n")
    outf.write("module load openmpi/intel/1.6.4\n")
    outf.write("module load cxxlibraries/boost/1.55.0-intel\n")
    outf.write("module load gcc/4.8.1\n")
    outf.write("module load hdf5/1813-v18-openmpi-intel\n")
    outf.write("module load use.own\n")

    outf.write("module load nengo\n\n")

    outf.write("cd %s\n" % directory)

    outf.write("cp ${PBS_NODEFILE} .\n\n")

    outf.write("# EXECUTION COMMAND;\n")
    outf.write(
        "mpirun -np {0} --mca pml ob1 {6} {1} {2} {3} {4} "
        "> {5}\n".format(
            num_processors, network_file, sim_time,
            show_prog, log, results, worker_location))

shutil.copy(network_file, directory)
os.chdir(directory)
job_id = subprocess.check_output(['qsub', submit_script_name])
job_id = job_id.split('.')[0]

open(directory + '/' + job_id, 'w').close()
print "Job ID: ", job_id
