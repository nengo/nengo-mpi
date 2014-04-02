import nengo_mpi
import numpy as np

x = nengo_mpi.PythonMpiSimulatorChunk()
y = np.array(np.arange(10.0))
z = np.random.random((5,5))
x.add_signal(id(y), y)
x.add_signal(id(z), z)
