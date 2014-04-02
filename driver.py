import nengo_mpi
import numpy as np

x = nengo_mpi.PythonMpiSimulatorChunk()
y = np.array(np.arange(10.0))
w = np.array(np.arange(10.0) + 1)
a = np.array(np.arange(5.0))
b = np.array(np.zeros(10))
z = np.random.random((10,5))

x.add_signal(id(y), y)
x.add_signal(id(z), z)
x.add_signal(id(w), w)
x.add_signal(id(a), a)
x.add_signal(id(b), b)

x.create_Reset(id(y), .1)
x.create_Copy(id(w), id(y))
x.create_Reset(id(w), .2)
x.create_DotInc(id(z), id(a), id(b))

x.run_n_steps(5)
