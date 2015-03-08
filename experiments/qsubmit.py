"""
A script for submitting nengo_mpi experiments to the GPC job queue.

Creates a new directory for each experiment in ``experiments_dir'',
which is labelled by date and time the experiment was started.
Launches the script from that directory, and then writes the output
from running the script to ``results.txt'' in that directory.
Also creates a sym link called ``latest'' in ``experiments_dir'',
which points to the directoy of the most recently run experiment.

Example:

For a simulation that uses 3 hardware nodes, 20 minutes of wallclock
time and gives the args ``-p 24 -d 5'' to the nengo script, run the
command:

python qsubmit.py -n 3 -w '0:20:00' -o '-p 24 -d 5'
"""

import string
import os
import subprocess
import argparse
import datetime

parser = argparse.ArgumentParser(
    description="Run nengo mpi simulation using GPC queue.")

parser.add_argument(
    '-N', type=int, default=0,
    help="Number of hardware nodes. If supplied, overrides "
         "the arguments to the nengo script as well.")

parser.add_argument(
    '-n', type=int, default=1,
    help="Number of hardware nodes. Same as -N, but doesn't "
         "override arguments to the nengo script.")

parser.add_argument(
    '-o', default='',
    help="Arguments for the nengo script. Should be in quotes.")

parser.add_argument(
    '-w', default="0:15:00",
    help="Upper bound on required wall time.")

parser.add_argument(
    '-d', action="store_true", default=False,
    help="Supply this arg to run an interactive debug session.")

args = parser.parse_args()

script_args = args.o

if args.N:
    num_nodes = args.N
    script_args = (
        "-p %d --sl %d " % (num_nodes * 8, num_nodes * 8) + script_args)
else:
    num_nodes = args.n

wall_time = args.w
debug = args.d

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

nengo_script_location = (
    '/home/c/celiasmi/e2crawfo/nengo_mpi/scripts/benchmark.py')


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

    line = "#PBS -l nodes=%d:ppn=8,walltime=%s " % (num_nodes, wall_time)
    if debug:
        line += "-I -X -qdebug"
    outf.write(line + "\n")

    outf.write("#PBS -N nengo_mpi\n")
    outf.write("#PBS -m abe\n\n")

    outf.write(
        "# load modules (must match modules used for compilation)\n")
    outf.write("module load intel/14.0.1\n")
    outf.write("module load python/2.7.5\n")
    outf.write("module load openmpi/intel/1.6.4\n")
    outf.write("module load cxxlibraries/boost/1.55.0-intel\n")
    outf.write("module load gcc/4.8.1\n")
    outf.write("module load use.own\n")

    if debug:
        outf.write("module load extras ddt\n")

    outf.write("module load nengo\n\n")

    outf.write("cd %s\n" % directory)

    outf.write("cp ${PBS_NODEFILE} .\n\n")

    outf.write("# EXECUTION COMMAND;\n")
    outf.write(
        "mpirun -np 1 --mca pml ob1 python %s --noprog %s "
        "> %s\n" % (nengo_script_location, script_args, results))

os.chdir(directory)
job_id = subprocess.check_output(['qsub', submit_script_name])
job_id = job_id.split('.')[0]

open(directory + '/' + job_id, 'w').close()
print "Job ID: ", job_id
