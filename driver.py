import nengo_mpi
import numpy as np

x = nengo_mpi.PythonMpiSimulatorChunk()
y = np.array(np.arange(10.0))

x.add_signal('a', y)
