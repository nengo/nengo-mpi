FAQ
---

**Is there any build step parallelization?**

No, nengo_mpi only provides parallelization for the simulation step. The build step
is where all the really difficult stuff happens, which, for instance, makes an Ensemble
act like an Ensemble. Therefore, nengo_mpi simply uses vanilla nengo's 
builder, which runs serially in python.

During an invocation such as: ::

    mpirun -np 8 python -m nengo_mpi nengo_script.py

the build step is performed entirely by the process with index 0.

It is definitely possible to create a parallelized version of the builder.
However, that should probably use a more python-friendly,
platform-agnostic technology than MPI (something like ZeroMQ). In other words,
thats another project.

======================

**What is the difference between
a cluster a component, a partition, a chunk, a process, a processor, and a node?
I've seen all these words used in the code with apparently similar meanings.**

All these terms do in fact have precise meanings in the context of nengo_mpi. They
can nicely be divided up into terms that apply at build time
and terms that apply at simulation time.

* **Build Time**

    * A ``cluster`` (distinct from a cluster of machines in high-performance computing)
      is a group of nengo objects that must be simulated together, for
      any of a number of reasons (see the class `NengoObjectCluster` in `partition/base.py`).
      The most prominent reason is that there is an
      path of Connections between the two objects that does not have a
      synapse (since synapses are the main source of "update" operators; see :ref:`how_it_works`).
      Another common reason is that the two objects are connected by a Connection
      which has a learning rule. The partitioning step applies a partitioning function
      to a graph whose nodes are ``clusters``.

    * A ``component`` (as in a component of a partition) is a group of ``clusters`` that
      will be simulated together. ``Components`` are computed by the partitioning step.
      When creating an instance of ``nengo_mpi.Simulator``, we typically specify the number
      of ``components`` that we want the network to be divided into. When nengo_mpi saves
      a network to file for communication with the C++ code, each ``component`` is
      stored separately.

    * A ``partition`` is a collection of ``components``. The goal of the partitioning step
      is to create a partition of the set of clusters, in the sense used
      `here <https://en.wikipedia.org/wiki/Partition_of_a_set>`_. High-quality partitions
      are those which do not assign drastically different amounts of work to different
      components, and which minimize the amount of communication between components.

* **Simulation Time**

    * A ``process`` is, of course, an OS abstraction for a line of computation. A ``processor``
      is a physical computation device. ``Processes`` run on ``processors``. It is generally
      possible to run a nengo_mpi simulation using more ``processes`` than there are ``processors``
      available on the machine, however the amount of
      parallelization we can obtain is determined by the number of physical
      ``processors`` (though hyperthreading can increase the effective number of ``processors``).
      The number of ``processes`` used to run a simulation is specified by the
      ``-np <NP>`` command-line argument when calling ``mpi_run``.

    * A ``chunk`` (see ``chunk.hpp``) is the C++ code's abstraction for a collection of nengo
      objects (actually, signals and operators corresponding to those objects) that are being
      simulated by a single ``process``. There is a one-to-one relationship between ``chunks``
      and ``processes``. One of the first things that each ``process`` does is create a ``chunk``.

    * The relationship between ``chunks/processes`` and ``components`` is as follows. At build time
      the network is divided into some specified number of ``components`` by partitioning. At simulation
      time, some specified number of ``chunks/processes`` will be active. ``Components`` are assigned to
      ``chunks/processes`` in a round-robin fashion. For example, if there are 4 ``chunks/processes``
      active and the network to simulate has 7 components, then ``process 0`` simulates components
      0 and 4, ``process 1`` simulates 1 and 5, etc. If the network instead had only 3 ``components``,
      then ``process 3`` would be left without anything to simulate, which is perfectly OK.

    * In the world of High-Performance Computing, a ``node`` (distinct from a nengo Node) is a physical
      computer consisting of some number of ``processors``. On the General Purpose Cluster there are 8
      processors per node and on Bluegene/Q there are 16 (that becomes 16 for GPC and 64 for BGQ once
      hyperthreading is taken into account). When running on one of these high-performance clusters,
      jobs are assigned computational resources in units of ``nodes`` rather than ``processors``.
