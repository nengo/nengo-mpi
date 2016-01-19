*********
nengo_mpi
*********

`nengo_mpi <https://github.com/e2crawfo/nengo_mpi>`_ is a C++/MPI backend for
`nengo <https://pythonhosted.org/nengo/index.html>`_, a python library
for building and simulating biologically realistic neural networks.
nengo_mpi makes it possible to run nengo simulations in parallel on
thousands of processors, and existing nengo scripts can be adapted to
make use of nengo_mpi with minimal effort.

With an MPI implementation installed on the system, nengo_mpi can be used
to run neural simulations in parallel using just a few lines of code: ::

    import nengo
    import nengo_mpi
    import numpy as np
    import matplotlib.pyplot as plt

    with nengo.Network() as net:
        sin_input = nengo.Node(output=np.sin)

        # A population of 100 neurons representing a sine wave
        sin_ens = nengo.Ensemble(n_neurons=100, dimensions=1)
        nengo.Connection(sin_input, sin_ens)

        # A population of 100 neurons representing the square of the sine wave
        sin_squared = nengo.Ensemble(n_neurons=100, dimensions=1)
        nengo.Connection(sin_ens, sin_squared, function=np.square)

        # View the decoded output of sin_squared
        squared_probe = nengo.Probe(sin_squared, synapse=0.01)

    partitioner = nengo_mpi.Partitioner(2)
    sim = nengo_mpi.Simulator(net, partitioner=partitioner)
    sim.run(5.0)

    plt.plot(sim.trange(), sim.data[squared_probe])
    plt.show()

There are 3 differences between this script and a script using the
reference implementation of nengo. First, we need to ``import nengo_mpi``.
Then we need to create a ``Partitioner`` object, which specifies how many
components the nengo network should be split up into (this corresponds
to the maximum number of distinct processors that can be used to run
the simulation). Finally, when creating the simulator we need to use
nengo_mpi's Simulator class, and we need to pass in the Partitioner instance.

The script can then be run in parallel using::

    mpirun -np 2 python -m nengo_mpi <script_name>

``sin_ens`` and ``sin_squared`` will be simulated on separate processors, with the
output of ``sin_ens`` being passed to the input of ``sin_squared`` every
time-step using MPI. After the simulation has completed, the probed results
from all processors are passed back to the main processor, so all probed data
can be accessed in the usual way (e.g. ``sim.data[squared_probe]``).

nengo_mpi is fully featured, supporting all aspects of Nengo Release 2.0.2.

.. toctree::
   :maxdepth: 2

   getting_started
   user_guide
   dev_guide
