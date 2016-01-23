.. _getting_started:

***************
Getting Started
***************

Installation
============

At the present time, nengo_mpi is only known to be usable on Linux.
Obtaining all nengo_mpi functionality requires a working
installation of MPI, and the most recent version of nengo
and all associated dependencies.

Basic installation
------------------

To install nengo_mpi, we use ``git``: ::

   git clone https://github.com/nengo/nengo_mpi.git
   cd nengo_mpi
   python setup.py develop --user

If you're using a ``virtualenv`` (recommended!) then you can omit the ``--user`` flag.
The last step is compile the ``mpi_sim`` C++ library, which contains most of the functionality.
To do this, ``cd`` into ``nengo_mpi/mpi_sim`` and type ``make``. If an MPI implementation
is present on the system, then this process be relatively straightforward.

Coming soon: More info on trouble-shooting the compilation process.


Adapting Existing Nengo Scripts
===============================

Existing nengo scripts can be adapted to make use of nengo_mpi by making
just a few small modifications. The most basic change that needs to be made
is importing nengo_mpi in addition to nengo, and then using the
``nengo_mpi.Simulator`` class in place of the ``Simulator`` class provided by nengo ::

     import nengo_mpi
     import nengo

     ... Code to build network ...

     sim = nengo_mpi.Simulator(network)
     sim.run(1.0)

     plt.plot(sim.trange(), sim.data[probe])

This will run a simulation using the nengo_mpi backend, but does not yet take
advantage of parallelization. However, even without parallelization, the
nengo_mpi backend can often be quite a bit faster than the reference
implementation (see our :ref:`benchmarks`) since it is a C++ library
wrapped by a thin python layer, whereas the reference implementation is pure
python.

Partitioning
------------
In order to have simulations run in parallel, we need a way of specifying
which nengo objects are going to be simulated on which processors. A
:class:`Partitioner` is the abstraction we use to do this specification.
The most basic information that a partitioner requires is the
number of components to split the network into. We can supply this
information when creating the partitioner, and then pass the partitioner to the
Simulator object: ::

    partitioner = nengo_mpi.Partitioner(n_components=8)
    sim = nengo_mpi.Simulator(network, partitioner=partitioner)
    sim.run(1.0)

The number of components we specify here acts as an upper bound on the effective
number of processors that can be used to run the simulation.

We can also specify a partitioning function, which accepts a graph
(corresponding to a nengo network) and a number of components, and returns
a python dictionary which gives, for each nengo object, the component it has been
assigned to. If no partitioning function is supplied, then a default
is used which simply assigns each component a roughly equal number of neurons.
A more sophisticated partitioning function (which has additional dependencies)
uses the `metis <http://glaros.dtc.umn.edu/gkhome/metis/metis/overview>`_
package to assign objects to components in a way that minimizes
the number of nengo Connections that straddle component boundaries. For example: ::

    partitioner = nengo_mpi.Partitioner(n_components=8, func=nengo_mpi.metis_partitioner)
    sim = nengo_mpi.Simulator(network, partitioner=partitioner)
    sim.run(1.0)

For small networks, we can also supply a dict mapping from nengo objects to component indices: ::

    model = nengo.Network()
    with model:
        A = nengo.Ensemble(n_neurons=50, dimensions=1)
        B = nengo.Ensemble(n_neurons=50, dimensions=1)
        nengo.Connection(A, B)

    assignments = {A: 0, B: 1}
    sim = nengo_mpi.Simulator(model, assignments=assignments)
    sim.run(1.0)

Note, though, that this does not scale well and should be reserved for toy networks/demos.

Running scripts
===============

To use the nengo_mpi backend without parallelization, scripts modified
as above can be run in the usual way ::

    python nengo_script.py

This will run serially, even if we have used a partitioner to specify that the
network be split up into multiple components. When a script is run, nengo_mpi
automatically detects how many MPI processes are active, and assigns
components to each process. In this case only one process (the master
process) is active, and all components will be assigned to it.

In order to get parallelization we need a slightly more complex invocation: ::

    mpirun -np NP python -m nengo_mpi nengo_script.py

where NP is the number of MPI processes to launch. Its fine if NP is not
equal to the number of components that the network is split into; if NP is
larger, then some MPI processes will not be assigned any component to
simulate, and if NP is smaller, some MPI processes will be assigned multiple
components to simulate.
