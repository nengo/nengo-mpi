import os
import argparse
import numpy as np
import subprocess
import shutil


def go(network, loc, n_copies, base_seed):
    if not os.path.isdir(loc):
        os.makedirs(loc)

    n_procs = [1, 2, 4, 8,]# 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    n_ensembles = [16, 64,]# 256, 1024, 4096]

    if network == "grid":
        command_template = (
            "python grid.py --sl {sl} --ns {ns} -d 4 --npd 50 "
            "--mpi 1 -p {n_procs} --save {save} --seed {seed} "
            "--pfunc metis")
        save_template = "grid_p{n_procs}_e{n_ensembles}_c{copy}.net"

        for i in n_procs:
            for j in n_ensembles:
                dir_name = "procs{procs}_ens{ens}".format(procs=i, ens=j)
                dir_name = os.path.join(loc, dir_name)

                if not os.path.isdir(dir_name):
                    os.makedirs(dir_name)

                if j >= i:
                    for k in range(n_copies):
                        sqrtj = int(np.sqrt(j))
                        save = save_template.format(
                            n_procs=i, n_ensembles=j, copy=k)

                        command = command_template.format(
                            sl=sqrtj, ns=sqrtj, n_procs=i,
                            save=os.path.join(dir_name, save),
                            seed=base_seed+k)

                        output = subprocess.check_output(command.split(' '))
                        print output


if __name__ == "__main__":

    nengo_mpi_loc = os.path.join(os.getenv("HOME"), "nengo_mpi")

    parser = argparse.ArgumentParser(
        description="Create network files for benchmarking nengo_mpi.")

    parser.add_argument(
        'network', default="grid",
        type=str, help='The type of network to create.')

    parser.add_argument(
        '--loc', type=str, default='/data/nengo_mpi_benchmarking/',
        help='Place to store the resulting networks.')

    parser.add_argument(
        '--n-copies', type=int, default=10,
        help='Number of copies of each network size to make.')

    parser.add_argument(
        '--seed', type=int, default=1000,
        help='Seed used to generate networks.')
    args = parser.parse_args()

    loc = args.loc

    if args.network == "clean":
        files = os.listdir(loc)
        for f in files:
            path = os.path.join(loc, f)
            try:
                shutil.rmtree(path)
            except OSError:
                pass

    else:
        loc = os.path.join(loc, args.network)
        go(args.network, loc, args.n_copies, args.seed)
