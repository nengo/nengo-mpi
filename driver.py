import nengo_mpi
import numpy as np

x = nengo_mpi.PythonMpiSimulatorChunk()
y = np.array([1])

x.add_signal('a', y)
