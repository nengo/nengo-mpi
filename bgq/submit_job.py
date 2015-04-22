import os
import os.path as path
import string
import shutil
import subprocess
import argparse
import datetime
import numpy as np

parser = argparse.ArgumentParser(
    description="Run nengo mpi simulation using BGQ runjob.")

parser.add_argument(
    '-p', type=int, default=1024,
    help="Number of processors.")

parser.add_argument(
    'network', default="", type=str,
    help="The network file to run.")

parser.add_argument(
    '--log', default="", type=str,
    help="Location to save results to.")

parser.add_argument(
    '-t', default=1.0, type=float,
    help="The simulation time.")

parser.add_argument(
    '-w', default="0:30:00",
    help="Upper bound on required wall time.")

parser.add_argument(
    '--rpn', default=16, type=int, help="MPI Ranks per node.")

args = parser.parse_args()
print args

ranks_per_node = args.rpn
n_processors = args.p
n_nodes = np.ceil(n_processors / ranks_per_node)

network_file = args.network
assert network_file, "Must supply a network file to run."

network_name = path.split(network_file)[-1]
network_name = path.splitext(network_name)[0]
print "Network name: ", network_name

t = args.t

log = args.log
if log:
    log = " --log " + log

wall_time = args.w

experiments_dir = path.join(os.getenv('SCRATCH'), "experiments")
working_dir = path.join(experiments_dir, network_name + "_")
date_time_string = str(datetime.datetime.now()).split('.')[0]
date_time_string = reduce(
    lambda y, z: string.replace(y, z, "_"),
    [date_time_string, ":", " ", "-"])
working_dir += date_time_string
working_dir += '_p_%d' % n_processors

if not path.isdir(working_dir):
    os.makedirs(working_dir)

submit_script = "submit_script.sh"


def make_sym_link(target, name):
    try:
        os.remove(name)
    except OSError:
        pass

    os.symlink(target, name)


exe_loc = path.join(os.getenv('HOME'), 'nengo_mpi/bin/nengo_mpi')

with open(path.join(working_dir, submit_script), 'w') as outf:
    outf.write('#!/bin/sh\n')
    outf.write('# @ job_name           = %s\n' % network_name)
    outf.write('# @ job_type           = bluegene\n')
    outf.write('# @ comment            = "BGQ Job By Size"\n')
    outf.write('# @ error              = $(job_name).$(Host).$(jobid).err\n')
    outf.write('# @ output             = $(job_name).$(Host).$(jobid).out\n')
    outf.write('# @ bg_size            = 64\n' % n_nodes)
    #outf.write('# @ bg_size            = %d\n' % n_nodes)
    outf.write('# @ wall_clock_limit   = %s\n' % wall_time)
    outf.write('# @ bg_connectivity    = Torus\n')
    outf.write('# @ queue \n')

    outf.write(
        'runjob --np {p} --ranks-per-node={ranks_per_node} '
        '--envs OMP_NUM_THREADS=1 HOME=$HOME --cwd={cwd} : '
        '{exe_loc} {log} {network_file} {t}'.format(
            p=n_processors, ranks_per_node=ranks_per_node, exe_loc=exe_loc,
            log=log, network_file=network_file, t=t, cwd=working_dir))

make_sym_link(working_dir, path.join(experiments_dir, 'latest'))

# shutil.copy(network_file, working_dir)
make_sym_link(
    path.join(os.getcwd(), network_file),
    path.join(working_dir, path.split(network_file)[-1]))

os.chdir(working_dir)

# Submit to loadleveler
job_id = subprocess.check_output(['llsubmit', submit_script])
print job_id