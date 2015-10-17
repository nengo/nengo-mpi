***************
Getting Started
***************

Installation
============

At the present time, ``nengo_mpi`` is only known to be usable on Linux.
Obtaining all ``nengo_mpi`` functionality requires a working
installation of MPI, and the most recent version of ``nengo``
and all associated dependencies.

However, ``nengo_mpi`` is also sub-divided into several modular components,
and some of these can be useful in some contexts without the other components.
TODO

Basic installation
------------------

To install nengo_mpi, we use ``git``: ::

   git clone https://github.com/nengo/nengo_mpi.git
   cd nengo_mpi
   python setup.py develop --user

If you're using a ``virtualenv``
(recommended!) then you can omit the ``--user`` flag.

Adapting Existing Nengo Scripts
===============================

Existing Nengo scripts can be adapted to make use of nengo_mpi by making
just a few small modifications. The most basic change that needs to be made
is importing nengo_mpi in addition to nengo, and then using the
nengo_mpi.Simulator class in place of the Simulator class provided by nengo ::

     import nengo_mpi
     import nengo

     ... Code to build network ...

     sim = nengo_mpi.Simulator(network)
     sim.run(1.0)

     plt.plot(sim.trange(), sim.data[probe])

This will run a simulation using the nengo_mpi backend, but does not yet take
advantage of any parallelization. However, even without parallelization, the
nengo_mpi backend can often be quite a bit faster than the reference
implementation (see our :ref:`benchmarks` for evidence).

Partitioners
------------
In order to have simulations run in parallel, we need a way of specifying
which nengo objects are going to be simulated on which processors. A
:class:`Partitioner` is an object that stores information on how to do this
specification. The most basic information that a partitioner requires is the
number of components to split the network into. We can supply this
information when creating the partitioner, and then pass the partitioner to the
Simulator object: ::

    partitioner = nengo_mpi.Partitioner(n_components=8)
    sim = nengo_mpi.Simulator(network, partitioner=partitioner)
    sim.run(1.0)

As we will see a bit later, the number of components we specify here acts as
an upper bound on the effective number of processors that can be used to run
the simulation.

We can also specify a partitioning function, which accepts a graph
(corresponding to a nengo network) and a number of components, and returns
dictionary which gives, for each nengo object, the component it has been
assigned to. If no partitioning function is supplied, then a default
partitioning function is used which simply attempts to assign each component
a roughly equal number of neurons. A more sophisticated partitioning function
uses the ``metis`` software library (which requires additional dependencies)
to attempt to assign objects to components in a way that minimizes the number
of nengo Connections that straddle component boundaries. For example ::

    partitioner = nengo_mpi.Partitioner(n_components=8, func=nengo_mpi.metis_partitioner)
    sim = nengo_mpi.Simulator(network, partitioner=partitioner)
    sim.run(1.0)

Running scripts
--------------------------
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

Next steps
==========

TODO
* If you're wondering how this works and you're not
  familiar with the Neural Engineering Framework,
  we recommend reading
  `this technical overview <http://compneuro.uwaterloo.ca/files/publications/stewart.2012d.pdf>`_.
* If you have some understanding of the NEF already,
  or just want to dive in headfirst,
  check out `our extensive set of examples <examples.html>`_.
* If you want to see the real capabilities of Nengo, see our
  `publications created with the NEF and Nengo <http://compneuro.uwaterloo.ca/publications.html>`_.
