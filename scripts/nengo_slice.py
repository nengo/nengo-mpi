
import logging

import numpy as np
import pytest

import nengo_mpi
import nengo
from nengo.connection import ConnectionSolverParam
from nengo.solvers import LstsqL2
from nengo.utils.functions import piecewise
from nengo.utils.numpy import filtfilt
from nengo.utils.testing import allclose

logger = logging.getLogger(__name__)

seed = 1
Simulator = nengo_mpi.Simulator
nl = nengo.LIFRate

N = 300

x = np.array([-1, -0.25, 1])

s1a = slice(1, None, -1)
s1b = [2, 0]
T1 = [[-1, 0.5], [2, 0.25]]
y1 = np.zeros(3)
y1[s1b] = np.dot(T1, x[s1a])

s2a = [0, 2]
s2b = slice(0, 2)
T2 = [[-0.5, 0.25], [0.5, 0.75]]
y2 = np.zeros(3)
y2[s2b] = np.dot(T2, x[s2a])

s3a = [2, 0]
s3b = [0, 2]
T3 = [0.5, 0.75]
y3 = np.zeros(3)
y3[s3b] = np.dot(np.diag(T3), x[s3a])

sas = [s1a, s2a, s3a]
sbs = [s1b, s2b, s3b]
Ts = [T1, T2, T3]
ys = [y1, y2, y3]

with nengo.Network(seed=seed) as m:
    m.config[nengo.Ensemble].neuron_type = nl()

    u = nengo.Node(output=x)
    a = nengo.Ensemble(N, dimensions=3, radius=1.7)
    nengo.Connection(u, a)
    pa = nengo.Probe(a, synapse=0.03)
    pu = nengo.Probe(u, synapse=0.03)

    probes = []
    for sa, sb, T in zip(sas, sbs, Ts):
        b = nengo.Ensemble(N, dimensions=3, radius=1.7)
        nengo.Connection(a[sa], b[sb], transform=T)
        probes.append(nengo.Probe(b, synapse=0.03))

sim = Simulator(m)
sim.run(0.2)
t = sim.trange()

print "pu: ", sim.data[pu]
print "pa: ", sim.data[pa]

atol = 0.01 if nl is nengo.Direct else 0.1
for i, [y, p] in enumerate(zip(ys, probes)):
    print "i:", i, "y:", y
    #print sim.data[p]
    #assert np.allclose(y, sim.data[p][-20:], atol=atol), "Failed %d" % i