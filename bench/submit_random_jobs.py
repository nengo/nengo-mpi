from __future__ import print_function
import os
import re
import subprocess
import time

loc = os.path.join(os.getenv("HOME"), "nengo_mpi/networks/random")
dest = os.path.join(os.getenv("SCRATCH"), "experiments/random")

print("Source: %s" % loc)
print("Dest: %s" % dest)

if not os.path.isdir(dest):
    os.makedirs(dest)

command_template = (
    "python ../submit_job.py {netfile} -p {nprocs} "
    "-t {t} --envs {envs} --dir %s" % dest)
pattern = "random_p(?P<p>.+)_e(?P<e>.+)_q(?P<q>.+)_c(?P<c>.+).net"

envs = "NENGO_MPI_RUNTIMES_FILE=%s NENGO_MPI_LOADTIMES_FILE=%s" % (
    os.path.join(dest, "runtimes.csv"), os.path.join(dest, "loadtimes.csv"))

max_procs = 1024
t = 5.0
n_repeats = 1

for d in os.listdir(loc):
    dir_path = os.path.join(loc, d)
    if not os.path.isdir(dir_path):
        continue

    for netfile in os.listdir(dir_path):
        m = re.match(pattern, netfile)
        if m:
            n_procs = int(m.groupdict()['p'])
            q = float(m.groupdict()['q'])

            print("** Submitting job for %s" % netfile)
            netfile_path = os.path.join(dir_path, netfile)
            command = command_template.format(
                nprocs=n_procs, netfile=netfile_path, t=t, envs=envs)
            output = subprocess.check_output(command.split())
            print(output)
            time.sleep(1.1)

        else:
            print("** Ignoring non-network %s" % netfile)
