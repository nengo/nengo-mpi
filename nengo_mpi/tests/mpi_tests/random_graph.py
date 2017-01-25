import nengo

import nengo_mpi
from nengo_mpi.tests.test_mpi import refimpl_results

import numpy as np


class DummyRequest(object):
    pass


request = DummyRequest()
request.param = nengo.LIF

m, refimpl_sim, sim_time = refimpl_results(request)

partitioner = nengo_mpi.Partitioner(4)
sim = nengo_mpi.Simulator(m, partitioner=partitioner)

sim.run(sim_time)

for p in m.probes:
    assert np.allclose(
        refimpl_sim.data[p], sim.data[p],
        atol=0.00001, rtol=0.00)

for p in m.probes:
    assert np.allclose(
        refimpl_sim.data[p], sim.data[p],
        atol=0.00001, rtol=0.00)
