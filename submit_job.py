"""
A script for submitting nengo_mpi jobs to the scheduling queue
on either scinet bgq or scinet gpc.

"""

import os
import os.path as path
import string
import subprocess
import argparse
import datetime
import math
import re


def write_gpc_jobfile(
        filename, n_procs, network_file, network_name,
        working_dir, wall_time, exe_loc, t, log, merged, timing,
        envs, ranks_per_node=8):

    network_file = path.split(network_file)[-1]

    with open(filename, 'w') as outf:
        outf.write("#!/bin/bash\n")
        outf.write("# MOAB/Torque submission script for SciNet GPC\n")
        outf.write("#\n")

        n_nodes = int(math.ceil(float(n_procs) / ranks_per_node))

        line = (
            "#PBS -l nodes=%d:ppn=%d,walltime=%s"
            "\n" % (n_nodes, ranks_per_node, wall_time))
        outf.write(line)
        outf.write("#PBS -N %s\n" % network_name)
        outf.write("#PBS -m abe\n")
        outf.write("#PBS -e %s.err\n" % network_name)
        outf.write("#PBS -o %s.out\n\n" % network_name)

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

        for e in envs:
            outf.write("export %s\n" % e)
        if envs:
            outf.write("\n")

        outf.write("cd %s\n" % working_dir)
        outf.write("cp ${PBS_NODEFILE} .\n\n")

        outf.write("# EXECUTION COMMAND;\n")
        outf.write(
            "mpirun -np {p} --mca pml ob1 {exe_loc} --noprog {log} {merged} "
            "{timing} {network_file} {t}\n".format(
                p=n_procs, ranks_per_node=ranks_per_node, exe_loc=exe_loc,
                log=log, merged=merged, timing=timing,
                network_file=network_file, t=t))


def write_bgq_jobfile(
        filename, n_procs, network_file, network_name,
        working_dir, wall_time, exe_loc, t, log, merged, timing,
        envs, ranks_per_node=16):

    n_nodes = math.ceil(float(n_procs) / ranks_per_node)
    n_nodes = int(max(64, n_nodes))

    network_file = path.split(network_file)[-1]
    with open(filename, 'w') as outf:
        outf.write('#!/bin/sh\n')
        outf.write('# @ job_name           = %s\n' % network_name)
        outf.write('# @ job_type           = bluegene\n')
        outf.write('# @ comment            = "nengo_mpi BGQ Job"\n')
        outf.write('# @ error              = %s.err\n' % network_name)
        outf.write('# @ output             = %s.out\n' % network_name)
        outf.write('# @ bg_size            = %s\n' % n_nodes)
        outf.write('# @ wall_clock_limit   = %s\n' % wall_time)
        outf.write('# @ bg_connectivity    = Torus\n')
        outf.write('# @ queue \n')

        outf.write(
            'runjob --np {p} --ranks-per-node={ranks_per_node} '
            '--envs OMP_NUM_THREADS=1 HOME=$HOME {envs} --cwd={wd} : '
            '{exe_loc} --noprog {log} {merged} {timing} '
            '{network_file} {t}'.format(
                p=n_procs, ranks_per_node=ranks_per_node, exe_loc=exe_loc,
                log=log, merged=merged, timing=timing,
                network_file=network_file,
                t=t, wd=working_dir, envs=' '.join(envs)))


def handle_non_cluster(
        n_procs, network_file, network_name,
        working_dir, exe_loc, t, log, merged, timing, ranks_per_node):

    old_dir = os.getcwd()
    os.chdir(working_dir)

    command = (
        "mpirun -np {p} --mca pml ob1 {exe_loc} --noprog {log} {merged} "
        "{timing} {network_file} {t}\n".format(
            p=n_procs, ranks_per_node=ranks_per_node, exe_loc=exe_loc,
            log=log, merged=merged, timing=timing,
            network_file=network_file, t=t))
    try:
        output = subprocess.check_output(
            command.split(), stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print e.output
        raise

    with open(network_name+'.out', 'w') as f:
        f.write(output)
    os.chdir(old_dir)
    return output


def make_directory_name(experiments_dir, network_name, add_date=True):
    if add_date:
        working_dir = path.join(experiments_dir, network_name + "_")
        date_time_string = str(datetime.datetime.now()).split('.')[0]
        date_time_string = reduce(
            lambda y, z: string.replace(y, z, "_"),
            [date_time_string, ":", " ", "-"])
        working_dir += date_time_string
    else:
        working_dir = path.join(experiments_dir, network_name)

    return working_dir


def make_sym_link(target, name):
    try:
        os.remove(name)
    except OSError:
        pass

    os.symlink(target, name)


def submit_job(
        network_file, n_procs, log, merged=False, timing=False, t=1.0,
        wall_time=None, rpn=0, envs=None, dir=None, add_date=True):

    platform = ''
    host_name = os.getenv("HOSTNAME")
    if host_name:
        if 'gpc' in host_name:
            platform = 'gpc'
        elif 'bgq' in host_name:
            platform = 'bgq'

    if not platform:
        print("Not on known cluster. Attempting to run the network, "
              "rather than submitting as a job.")

    if rpn == 0:
        ranks_per_node = {'gpc': 8, 'bgq': 16, '': 8}[platform]
    else:
        ranks_per_node = rpn

    if log:
        log = " --log " + log

    merged = "--merged" if merged else ""
    timing = "--timing" if timing else ""

    if wall_time is None:
        wall_time = {
            'gpc': '0:15:00', 'bgq': '0:30:00', '': '0:15:00'}[platform]
    else:
        wall_time = wall_time

    # Checks envs - \S matches any non-whitespace character
    envs = [] if envs is None else envs
    pattern = '\S+=\S+'
    for e in envs:
        assert re.match(pattern, e), "``envs`` supplied in incorrect format."

    # Define constants
    exe_loc = path.join(os.getenv('HOME'), 'nengo_mpi/bin/nengo_mpi')
    experiments_dir = (
        path.join(os.getenv('SCRATCH'), "experiments")
        if dir is None else dir)

    network_name = path.basename(network_file)
    network_name = path.splitext(network_name)[0]
    print "Network name: ", network_name

    # Create directory to run the job from
    working_dir = make_directory_name(
        experiments_dir, network_name, add_date=add_date)

    if not path.isdir(working_dir):
        os.makedirs(working_dir)

    # Create convenience `latest` symlink
    make_sym_link(working_dir, path.join(experiments_dir, 'latest'))

    # Make a symlink in the working dir to the
    # file containing the network that we want to load and simulate
    make_sym_link(
        path.join(os.getcwd(), network_file),
        path.join(working_dir, path.basename(network_file)))

    if platform not in ['gpc', 'bgq']:
        if envs:
            raise Exception(
                "Cannot supply environment variables on non-cluster. Instead, "
                "set environment variables in calling shell.")
        handle_non_cluster(
            n_procs, path.basename(network_file), network_name,
            working_dir, exe_loc, t, log, merged, timing, ranks_per_node)
        return

    submit_script = "submit_script.sh"
    submit_script_path = path.join(working_dir, submit_script)

    # Create the job file in the working dir
    if platform == 'gpc':
        write_gpc_jobfile(
            submit_script_path, n_procs, network_file, network_name,
            working_dir, wall_time, exe_loc, t, log, merged, timing,
            envs, ranks_per_node)
    elif platform == 'bgq':
        write_bgq_jobfile(
            submit_script_path, n_procs, network_file, network_name,
            working_dir, wall_time, exe_loc, t, log, merged, timing,
            envs, ranks_per_node)
    else:
        raise NotImplementedError()

    os.chdir(working_dir)

    # Submit to queue
    if platform == 'gpc':
        job_id = subprocess.check_output(['qsub', submit_script])
        job_id = job_id.split('.')[0]
    elif platform == 'bgq':
        job_id = subprocess.check_output(['llsubmit', submit_script])
        job_id = job_id.split('"')[1]
    else:
        raise NotImplementedError()

    # Create a file in the directory with the job_id as its name
    open(job_id, 'w').close()
    print "Job ID: ", job_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Submit a nengo_mpi simulation using either "
                    "BGQ llsubmit or GPC qsub.")

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
        '--merged', action="store_true",
        help="Supply to run using merged mpi communication.")

    parser.add_argument(
        '--timing', action="store_true",
        help="Supply to collect timing information.")

    parser.add_argument(
        '-t', default=1.0, type=float,
        help="The simulation time.")

    parser.add_argument(
        '-w', default=None,
        help="Upper bound on required wall time.")

    parser.add_argument(
        '--rpn', default=0, type=int, help="MPI Ranks per node.")

    parser.add_argument(
        '--envs', default=None, nargs='*',
        help="Environment variables to run the job with.")

    parser.add_argument(
        '--dir', default=None,
        help="Path to store the experiment directories.")

    parser.add_argument(
        '--add-date', action='store_true',
        help="Whether to add date to experiment directory names.")

    # Parse args
    args = parser.parse_args()
    print args

    submit_job(
        args.network, args.p, args.log, args.merged, args.timing, args.t,
        args.w, args.rpn, args.envs, args.dir, args.add_date)
