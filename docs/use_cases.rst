Use Cases
=========

There are two distinct ways to use nengo\_mpi.

1. Build With Python, Simulate From Python
------------------------------------------

This is the most straightforward way to run simulations. Existing nengo scripts can quickly be adapted to use nengo_mpi using this method.

2. Build With Python, Simulate Using Executable
-----------------------------------------------

Using this technique, the process of building networks is similar to the first method, while the process of running the simulations is is a bit different. However, this approach offers more flexibility, allowing simulations to be built on one computer with full python support, and then simulated on another computer that has greater parallel computation capabilities (e.g. a cluster).  One thing to be aware of with this method is that it has some limitations. In particular, this technique cannot deal with non-trivial nengo Nodes. The reason is that at simulation time, python will be completely out of the picture, so the arbitrary code that can be contained in the Nodes cannot be executed. It is possible that this could be fixed in the future by spinning up a python interpreter at simulation time, though this would involve a significant amount of work. At the present time, the only nengo Nodes that can be managed are passthrough nodes, nodes that output a constant signal, and Spaun Stimulus nodes, which we have made special accomodations for.

The first step is to build a network. To do this, we need to make a change to how we call nengo_mpi.Simulator. In particular, we supply the ``save_file`` argument: ::

    sim = nengo_mpi.Simulator(model, partitioner=partitioner, save_file="model.net")

This call will create a file called ``model.net`` in the current directory, which stores the operators and signals required to simulate the nengo Network specified by `model``. This file will actually be an HDF5 file , but we typically give it the ``.net`` extension to indicate that it stores a built network. Another point of difference from the previous method is that the script can be executed using a simpler invocation ::

    python nengo_script.py

In fact, we do not need MPI installed on the system at all to create network files like this, and the C++ code does not need to be compiled.

Now we can make use of the network file we've created using the ``nengo_mpi`` executable. Assuming now that MPI *is* installed and the C++ code *has* been compiled, we can run: ::

    mpirun -np NP nengo_mpi model.net 1.0

where NP is the number of MPI processors to use. This will simulate the network stored in ``model.net`` for 1 second of simulation time. Note that the the network has to have been split up into at least NP components at build time using a Partitioner object in order for all NP processes to be used.

The result of the simulation (the data collected by the probes) will be stored in an HDF5 file called ``model.h5``. We can specify a different name for the output file as follows: ::

    mpirun -np NP nengo_mpi --log results.h5 model.net 1.0

Finally, if MPI is not available on the system, we can instead use: ::

    nengo_cpp --log results.h5 model.net 1.0

but this will, of course, run serially.
