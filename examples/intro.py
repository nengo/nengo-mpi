# The example from the main page of the nengo_mpi docs:
#     https://nengo-mpi.readthedocs.io/en/readthedocs/index.html
#
# To run with parallelization, invoke using:
#     mpirun -np 2 python -m nengo_mpi intro.py
#
# To run serially, invoke using:
#     python intro.py
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
