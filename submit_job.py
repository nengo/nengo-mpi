"""
A script for submitting nengo_mpi jobs to the scheduling queue on either
scinet bgq or scinet gpc.
"""


import os
import os.path as path
import string
import subprocess
import argparse
import datetime
import numpy as np


def write_bgq_jobfile(
        filename, n_processors, network_file, network_name,
        working_dir, wall_time, exe_loc, t, log, ranks_per_node=16):

    network_file = path.split(network_file)[-1]
    with open(filename, 'w') as outf:
        outf.write('#!/bin/sh\n')
        outf.write('# @ job_name           = %s\n' % network_name)
        outf.write('# @ job_type           = bluegene\n')
        outf.write('# @ comment            = "nengo_mpi BGQ Job"\n')
        outf.write('# @ error              = $(job_name).$(Host).$(jobid).err\n')
        outf.write('# @ output             = $(job_name).$(Host).$(jobid).out\n')
        outf.write('# @ bg_size            = 64\n')
        outf.write('# @ wall_clock_limit   = %s\n' % wall_time)
        outf.write('# @ bg_connectivity    = Torus\n')
        outf.write('# @ queue \n')

        outf.write(
            'runjob --np {p} --ranks-per-node={ranks_per_node} '
            '--envs OMP_NUM_THREADS=1 HOME=$HOME --cwd={wd} : '
            '{exe_loc} {log} {network_file} {t}'.format(
                p=n_processors, ranks_per_node=ranks_per_node, exe_loc=exe_loc,
                log=log, network_file=network_file, t=t, wd=working_dir))


def write_gpc_jobfile(
        filename, n_processors, network_file, network_name,
        working_dir, wall_time, exe_loc, t, log, ranks_per_node=8):

    network_file = path.split(network_file)[-1]

    with open(filename, 'w') as outf:
        outf.write("#!/bin/bash\n")
        outf.write("# MOAB/Torque submission script for SciNet GPC\n")
        outf.write("#\n")

        n_nodes = np.ceil(float(n_processors) / ranks_per_node)

        line = (
            "#PBS -l nodes=%d:ppn=%d,walltime=%s"
            "\n" % (n_nodes, ranks_per_node, wall_time))
        outf.write(line)

        outf.write("#PBS -N %s\n" % network_name)
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

        outf.write("cd %s\n" % working_dir)

        outf.write("cp ${PBS_NODEFILE} .\n\n")

        outf.write("# EXECUTION COMMAND;\n")
        outf.write(
            "mpirun -np {p} --mca pml ob1 {exe_loc} {log} {network_file} {t}"
            "\n".format(
                p=n_processors, ranks_per_node=ranks_per_node, exe_loc=exe_loc,
                log=log, network_file=network_file, t=t))


def make_directory_name(experiments_dir, network_name):
    working_dir = path.join(experiments_dir, network_name + "_")
    date_time_string = str(datetime.datetime.now()).split('.')[0]
    date_time_string = reduce(
        lambda y, z: string.replace(y, z, "_"),
        [date_time_string, ":", " ", "-"])
    working_dir += date_time_string
    working_dir += '_p_%d' % n_processors

    return working_dir


def make_sym_link(target, name):
    try:
        os.remove(name)
    except OSError:
        pass

    os.symlink(target, name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Submit a nengo_mpi simulation using either "
                    "BGQ llsubmit or GPC qsub.")

    parser.add_argument(
        'platform', type=str,
        help="The platform we're running on. Must be either 'bgq' or 'gpc'.")

    parser.add_argument(
        'network', default="", type=str,
        help="The network file to run.")

    parser.add_argument(
        '-p', type=int, default=1024,
        help="Number of processors.")

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
        '--rpn', default=0, type=int, help="MPI Ranks per node.")

    # Parse args
    args = parser.parse_args()
    print args

    platform = args.platform.lower()
    assert platform == 'gpc' or platform == 'bgq', (
        "Platform must be either 'gpc' or 'bgq'")

    network_file = args.network
    assert network_file, "Must supply a network file to run."

    n_processors = args.p
    if args.rpn == 0:
        ranks_per_node = {'gpc': 8, 'bgq': 16}[platform]
    else:
        ranks_per_node = args.rpn

    t = args.t

    log = args.log
    if log:
        log = " --log " + log

    wall_time = args.w

    # Define constants
    submit_script = "submit_script.sh"
    exe_loc = path.join(os.getenv('HOME'), 'nengo_mpi/bin/nengo_mpi')
    experiments_dir = path.join(os.getenv('SCRATCH'), "experiments")

    network_name = path.split(network_file)[-1]
    network_name = path.splitext(network_name)[0]
    print "Network name: ", network_name

    # Create directory to run the job from
    working_dir = make_directory_name(experiments_dir, network_name)
    if not path.isdir(working_dir):
        os.makedirs(working_dir)

    submit_script_path = path.join(working_dir, submit_script)

    # Create the job file in the working dir
    if platform == 'gpc':
        write_gpc_jobfile(
            submit_script_path, n_processors, network_file, network_name,
            working_dir, wall_time, exe_loc, t, log, ranks_per_node)
    else:
        write_bgq_jobfile(
            submit_script_path, n_processors, network_file, network_name,
            working_dir, wall_time, exe_loc, t, log, ranks_per_node)

    # Create convenience `latest` symlink
    make_sym_link(working_dir, path.join(experiments_dir, 'latest'))

    # Make a symlink in the working dir to the
    # file containing the network to simulate
    make_sym_link(
        path.join(os.getcwd(), network_file),
        path.join(working_dir, path.split(network_file)[-1]))

    os.chdir(working_dir)

    # Submit to queue
    if platform == 'gpc':
        job_id = subprocess.check_output(['qsub', submit_script])
        job_id = job_id.split('.')[0]
    else:
        job_id = subprocess.check_output(['llsubmit', submit_script])
        job_id = job_id.split('"')[1]

    # Create a file in the directory with the job_id as its name
    open(job_id, 'w').close()
    print "Job ID: ", job_id
