.. _workflows:

Workflows
=========

There are two distinct ways to use nengo_mpi.

1. Build With Python, Simulate From Python
------------------------------------------

This is the most straightforward way to run simulations. Existing nengo
scripts can quickly be adapted to use nengo_mpi with this method. This
workflow is described in :ref:`getting_started`.

2. Build With Python, Simulate Using Stand-Alone Executable
-----------------------------------------------------------

Using this workflow, the process of building networks is similar to the first
workflow, while the process of running the simulations is quite different. This
approach offers more flexibility, allowing simulations to be built on one
computer (which we'll call the "build" machine) with full python support
but no MPI installation, and then simulated on another computer (which we'll
call the "sim" machine) with a full MPI installation but no python support
(e.g. a cluster).

One point to be aware of with this method is that it has some limitations. In
particular, it cannot deal with networks containing non-trivial nengo Nodes. The
reason is that at simulation time, python will be completely out of the
picture, so there is no way to execute the python code that Nodes
contain. It is possible that this could be fixed in the future by spinning up
a python interpreter at simulation time, though this would involve at
significant amount of work. At the present time, the only nengo Nodes are
allowed are passthrough nodes, nodes that output a constant signal, and
SpaunStimulus nodes. The first two are trivial to implement, and the third we have
made special accommodations for.

Building
********

The first step is to build a network and save it to a file. To do this, we need
to make a change to how we call ``nengo_mpi.Simulator``. In particular, we supply
the ``save_file`` argument: ::

    sim = nengo_mpi.Simulator(model, partitioner=partitioner, save_file="model.net")

This call will create a file called ``model.net`` in the current directory,
which stores the operators and signals required to simulate the nengo Network
specified by ``model``. This file will actually be an HDF5 file, but we
typically give it the ``.net`` extension to indicate that it stores a built
network. The script can then be executed (on the "build" machine) using a simple
invocation: ::

    python nengo_script.py

Simulating
**********

Now we can make use of the network file we've created using the ``nengo_mpi``
executable (see :ref:`modules` for more info on the executable). Assuming that
we are now on the "sim" machine, and that the ``nengo_mpi`` executable has been
compiled, we can run: ::

    mpirun -np NP nengo_mpi model.net 1.0

where NP is the number of MPI processors to use. This will simulate the
network stored in ``model.net`` for 1 second of simulation time.

The result of the simulation (the data collected by the probes) will be stored
in an HDF5 file called ``model.h5``. We can specify a different name for the
output file as follows: ::

    mpirun -np NP nengo_mpi --log results.h5 model.net 1.0

Finally, if MPI is not available on the "sim" machine, we can instead use: ::

    nengo_cpp --log results.h5 model.net 1.0

but this will run serially.
