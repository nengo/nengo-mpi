import nengo_mpi
import numpy as np

dt = 0.001
x = nengo_mpi.PythonMpiSimulatorChunk(dt)
y = np.array(np.arange(10.0))
w = np.array(np.arange(10.0) + 1)
a = np.array(np.arange(5.0))
b = np.array(np.zeros(10))
z = np.random.random((10,5))

lif_dest = np.zeros(10)
lif_rate_dest = np.zeros(10)

c = np.array(np.zeros(10))
d = np.array(np.zeros(10))
e = np.array(np.zeros(10))
k = np.array(np.zeros(10))
l = np.array(np.zeros(10))

x.add_signal(id(y), y)
x.add_signal(id(z), z)
x.add_signal(id(w), w)
x.add_signal(id(a), a)
x.add_signal(id(b), b)
x.add_signal(id(c), c)
x.add_signal(id(d), d)
x.add_signal(id(e), e)
x.add_signal(id(k), k)
x.add_signal(id(l), l)
x.add_signal(id(lif_dest), lif_dest)
x.add_signal(id(lif_rate_dest), lif_rate_dest)

x.create_Reset(id(y), .1)
x.create_Copy(id(w), id(y))
x.create_Reset(id(w), .2)
x.create_DotInc(id(z), id(a), id(b))
x.create_ProdUpdate(id(z), id(a), id(e), id(l))

count = 0
base = np.array(np.arange(10.0))
def f():
    global count
    count+=1;
    return base + count

def g(a):
    global count
    print a
    count+=1
    return a + count

def h(t):
    print "time:",t
    return t + np.zeros(10)

def i(t, inp):
    print "time:",t
    print "input:",inp
    return t + inp

x.create_PyFunc(id(e), h, True);
x.create_PyFunc(id(c), f, False);
x.create_PyFuncWithInput(id(d), g, False, id(c), c);
x.create_PyFuncWithInput(id(k), i, True, id(e), e);

x.create_SimLIF(10, 0.02, 0.002, dt, id(b), id(lif_dest))
x.create_SimLIFRate(10, 0.02, 0.002, dt, id(b), id(lif_rate_dest))

x.run_n_steps(10)

