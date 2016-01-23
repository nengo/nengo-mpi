.. _benchmarks:

Benchmarks
==========
Benchmarks testing the simulation speed of nengo_mpi were performed with 3
different machines and 3 different large-scale spiking neural networks. The
machines used were a home PC with a Quad-Core 3.6GHz Intel i7-4790 and 16 GB
of RAM, Scinet's `General Purpose Cluster
<https://wiki.scinet.utoronto.ca/wiki/index.php/GPC_Quickstart#Specifications>`_,
and Scinet's 4 rack
`Blue Gene/Q <https://wiki.scinet.utoronto.ca/wiki/index.php/BGQ#Specifications>`_.
We tested nengo_mpi using different numbers of processors, and also tested the
`reference implementation <https://github.com/nengo/nengo/tree/master/nengo>`_
of nengo (on the home PC only) for comparison.

Stream Network
--------------
The stream network exhibits a simple connectivity structure, and is intended
to be close to the optimal configuration for executing a simulation quickly In
parallel using nengo_mpi. In particular, the ratio of the amount of
communication vs computation per step is relatively low. The network takes a
single parameter :math:`n` giving the total number of neural ensembles. The
network contains :math:`\sqrt{n}` different "streams", where a stream is a
collection of :math:`\sqrt{n}` ensembles connected in a circular fashion (so
each ensemble has 1 incoming and 1 outgoing connection). Each ensemble is
:math:`4`-dimensional and contains :math:`200` LIF neurons, and we vary
:math:`n` as the independent variable in the graphs below. The largest network
contains :math:`2^{12} = 4096` ensembles, for a total of
:math:`200 * 4096 = 819,200` neurons. In every case, the ensembles are
distributed evenly amongst the processors. Each execution consists of 5
seconds of simulated time, and each data point is the result of 5 separate
executions.

.. image :: images/pc_stream_run.svg
.. image :: images/gpc_stream_run.svg
.. image :: images/bgq_stream_run.svg

Random Graph Network
--------------------
The random graph network is constructed by choosing a fixed number of
ensembles, and then randomly choosing ensemble-to-ensemble connections to
insantiate until a desired proportion of the total number of possible
connections is reached. In all cases, we use :math:`1024` ensembles, and we
vary the proportion of connections. This network is intended to show how the
performance of nengo_mpi scales as the ratio of communication to computation
increases, and investigate whether it is always a good idea to add more
processors. Adding more processors typically increases the amount of
inter-processor communication, since it increases the likelihood that any two
ensembles are simulated on different processors. Therefore, if communication
is the bottleneck, then adding more processors will tend to decrease performance.

Each ensemble is 2-dimensional and contains 100 LIF neurons, and each
connection computes the identity function. With :math:`n` ensembles, there are
:math:`n^2` possible connections (since we allow self-connections and the
connections are directed). Therefore in the most extreme case we have have
:math:`0.2 \times 1024^2 \approx 209,715` connections, each relaying a
2-dimensional value. The number of such connections that need to engage in
inter-processor communication is a function of the number of processors used
in the simulation, and the particular random connectivity structure that
arose. Each execution consists of 5 seconds of simulated time, and each data
point is the average of executions on 5 separate networks with different
randomly chosen connectivity structure.

.. image :: images/pc_random_run.svg
.. image :: images/gpc_random_run.svg
.. image :: images/bgq_random_run.svg

SPAUN
-----
SPAUN (Semantic Pointer Architecture Unified Network) is a large-scale,
functional model of the human brain developed at the CNRG. It is composed
entirely of spiking neurons, and can perform eight separate cognitive tasks
without modification. The original paper can be found `here
<http://compneuro.uwaterloo.ca/files/publications/eliasmith.2012.pdf>`_. SPAUN
is an extremely large-scale spiking neural network (currently approaching 4 million neurons
when using 512-dimensional semantic pointers) with very complex connectivity,
and represents a somewhat more realistic test than the more contrived examples used above.
In the plots below we vary the dimensionality of the semantic pointers, the internal
representations used by SPAUN.

.. image :: images/pc_spaun_run.svg
.. image :: images/gpc_spaun_run.svg
.. image :: images/bgq_spaun_run.svg

Clearly the larger clusters are providing less of a benefit here. The hypothesized reason
is that thus far we have been unable to split SPAUN up into sufficiently small
components than can be simulated independently. There are some components with many thousands
of neurons on them. Thus the limiting factor for the speed of simulation is how
quickly an individual processor is able to simulate one of these large components.
We can likely remedy this by playing around with the connectivity of SPAUN and finding
ways to reduce the maximum component size.

