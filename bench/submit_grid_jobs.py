from __future__ import print_function
import os
import re
import subprocess
import time

network = "grid"

loc = os.path.join(os.getenv("HOME"), "nengo_mpi/networks/grid")
dest = os.path.join(os.getenv("SCRATCH"), "experiments/grid")

print("Source: %s" % loc)
print("Dest: %s" % dest)

if not os.path.isdir(dest):
    os.makedirs(dest)

command_template = (
    "python ../submit_job.py {netfile} -p {nprocs} "
    "-t {t} --envs {envs} --dir %s" % dest)
pattern = "{0}_p(?P<p>.+)_e(?P<e>.+)_c(?P<c>.+).net".format(network)

envs = "NENGO_MPI_RUNTIMES_FILE=%s NENGO_MPI_LOADTIMES_FILE=%s" % (
    os.path.join(dest, "runtimes.csv"), os.path.join(dest, "loadtimes.csv"))

max_procs = 4096
t = 5.0
n_repeats = 5

for d in os.listdir(loc):
    dir_path = os.path.join(loc, d)
    if not os.path.isdir(dir_path):
        continue

    for netfile in os.listdir(dir_path):
        m = re.match(pattern, netfile)
        if m:
            n_procs = int(m.groupdict()['p'])

            if n_procs <= max_procs:
                for i in range(n_repeats):
                    print("** Submitting job for %s" % netfile)
                    netfile_path = os.path.join(dir_path, netfile)
                    command = command_template.format(
                        nprocs=n_procs, netfile=netfile_path, t=t, envs=envs)
                    output = subprocess.check_output(command.split(' '))
                    print(output)
                    time.sleep(1.1)
            else:
                print("** Ignoring %s because maxprocs (%s) "
                      "exceeded" % (netfile, max_procs))

        else:
            print("** Ignoring non-network %s" % netfile)
