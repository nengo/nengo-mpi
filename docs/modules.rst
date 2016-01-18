.. _modules:

=======
Modules
=======

nengo_mpi is composed of several fairly separable modules.

python
******

The python code consists primarily of alternate implementations of both ``nengo.Simulator`` and ``nengo.Model``. ``nengo_mpi.Simulator`` allows the C++/MPI code (detailed below) to be used in a manner nearly identical to ``nengo.Simulator`` (see :ref:`getting_started`). The ``nengo_mpi.Model`` class is primarily responsible for adapting the output of the reference implementation's build step to work with ``nengo_mpi.Simulator``.

C++
***

The directory mpi_sim contains the C++ code. This code implements a nengo simulator which can use MPI to run simulations in parallel. The C++ code only implements simulation capabilities; the *build* step (converting from a high-level model specification in terms of ensembles, connections, nodes and probes to a concrete computation graph) is still done in python, and nengo_mpi largely uses the builder provided by the reference implementation. The C++ code can be used in at least three different ways.

mpi_sim.so
----------
A shared library that allows the python layer to access the core C++ simulator. The python code creates an HDF5 file encoding the built network (the operators and signals that need to be simulated), and then makes a call out to ``mpi_sim.so`` with the name of the file. ``mpi_sim.so`` then opens the file and runs the simulation. The results are passed back to the python layer. The C++ simulator can thus be used in all the same ways as the reference implementation.

nengo_mpi executable
--------------------
This is an executable that allows the C++ simulator to be used directly, instead of having to go through python. The executable accepts as arguments the name of a file specifying the operators and signals in a network, as well as the length of time to run the simulation for, in seconds. Removing the requirement that the C++ code be accessed through python has a number of advantages. In particular, it can make attaching a debugger much easier. Also, some high-performance clusters (e.g. BlueGeneQ) provide only minimal support for python. The nengo_mpi exectuable has no python dependencies, and so it can be used on these machines. A typical usage pattern is to use the python code to create the HDF5 file on a machine with full python support, and then transfer that file over to the high-performance cluster where the network encoded by that file can be simulated using the nengo_mpi executable. See :ref:`workflows` for further details.

nengo_cpp executable
--------------------
This is simply a version of the nengo_mpi executable which does not have MPI dependencies (which, of course, means that there is no parallelization). It is possible that some users may find this useful in some situations where the MPI dependency cannot be met, as the C++ simulator is often significantly faster then the reference implementation simulator even without parallelization (see our :ref:`benchmarks`).
