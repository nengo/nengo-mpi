""" Setup mpi context.

The worker processes enter the C++ code and wait for signals from the master
process. The master process executes the script given as the first argument.

Example usage: mpirun -np <np> python -m nengo_mpi <script>

This code is only executed if nengo_mpi is run as the main script with -m.

"""


def main():
    """ Needs to be inside a function, because we are clearing the global
        namespace further down.

    """
    import sys
    import os
    import atexit
    from six import print_

    # cargo cult to get around errors. See
    # https://xrunhprof.wordpress.com/2014/11/04/an-openmpi-python-and-dlopen-issue/
    import ctypes
    ctypes.CDLL("libmpi.so", mode=ctypes.RTLD_GLOBAL)

    native_sim = ctypes.CDLL("mpi_sim.so")

    native_sim.init()

    rank = native_sim.get_rank()
    n_procs = native_sim.get_n_procs()

    if rank == 0 and (not sys.argv[1:] or sys.argv[1] in ("--help", "-h")):
        print_("usage: mpirun -np <np> "
               "python -m nengo_mpi scriptfile [arg] ...")
        sys.exit(2)

    if rank > 0:
        native_sim.worker_start()
    else:
        # Note: Largely copied from /usr/lib/python2.7/pdb.py
        mainpyfile = sys.argv[1]     # Get script filename
        if not os.path.exists(mainpyfile):
            print_('Error:', mainpyfile, 'does not exist')
            sys.exit(1)

        del sys.argv[0]         # Hide "pdb.py" from argument list

        # Replace pdb's dir with script's dir in front of module search path.
        sys.path[0] = os.path.dirname(mainpyfile)

        try:
            # Here we clear the global namespace and add special variables.
            # We need to use the current global namespace as the global
            # namespace for the script mainpyfile, otherwise any code that
            # makes use of ``import __main__`` will break.
            g = globals()
            builtins = g['__builtins__']
            g.clear()
            g.update({
                "__name__": "__main__",
                "__file__": mainpyfile,
                "__builtins__": builtins})

            with open(mainpyfile) as f:
                code = compile(f.read(), mainpyfile, 'exec')
                exec(code, g)

        except SystemExit:
            print_("The program exited via sys.exit(). Exit status: ",)
            print_(sys.exc_info()[1])

        # Closes all nengo_mpi Simulator instances, since
        # nengo_mpi.Simulator.close_simulators is an atexit exitfunc.
        atexit._run_exitfuncs()

        native_sim.kill_workers()

    native_sim.finalize()

main()
