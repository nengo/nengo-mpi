import nengo
import nengo_mpi
import pytest

network = nengo.Network()

with network:
    node = nengo.Node(0.5)
    ens1 = nengo.Ensemble(100, 1)
    ens2 = nengo.Ensemble(100, 1)

    nengo.Connection(node, ens1, synapse=0.01)
    nengo.Connection(ens1, ens2, synapse=0.01)
    nengo.Probe(ens2, synapse=0.01)

partitioner = nengo_mpi.Partitioner(2)
sim = nengo_mpi.Simulator(network, partitioner=partitioner)
sim.run(0.1)
with pytest.raises(RuntimeError) as e:
    sim = nengo_mpi.Simulator(network)

assert str(e.value).startswith("Attempting to create")

sim.close()
assert nengo_mpi.Simulator.all_closed()
