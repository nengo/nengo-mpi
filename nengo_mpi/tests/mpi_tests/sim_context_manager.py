import nengo
import nengo_mpi

network = nengo.Network()

with network:
    node = nengo.Node(0.5)
    ens1 = nengo.Ensemble(100, 1)
    ens2 = nengo.Ensemble(100, 1)

    nengo.Connection(node, ens1, synapse=0.01)
    nengo.Connection(ens1, ens2, synapse=0.01)
    nengo.Probe(ens2, synapse=0.01)

assert nengo_mpi.Simulator.all_closed()
partitioner = nengo_mpi.Partitioner(2)

with nengo_mpi.Simulator(network, partitioner=partitioner) as mpi_sim:
    assert not nengo_mpi.Simulator.all_closed()
    mpi_sim.run(0.1)

assert nengo_mpi.Simulator.all_closed()

mpi_sim = nengo_mpi.Simulator(network)
mpi_sim.run(0.1)
assert not nengo_mpi.Simulator.all_closed()
mpi_sim.close()

assert nengo_mpi.Simulator.all_closed()

with nengo_mpi.Simulator(network, partitioner=partitioner) as mpi_sim:
    assert not nengo_mpi.Simulator.all_closed()
    mpi_sim.run(0.1)

assert nengo_mpi.Simulator.all_closed()
