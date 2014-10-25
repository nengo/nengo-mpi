import logging
import pytest

import nengo
import nengo_mpi

logger = logging.getLogger(__name__)


def test_serialization():
    """Test the serialization of MPISimulators"""

    N = 30
    val = 0.5

    network = nengo.Network(label='simple', seed=123)
    with network:
        input = nengo.Node(output=val, label='input')
        A = nengo.Ensemble(n_neurons=N, dimensions=1, label='A')
        B = nengo.Ensemble(n_neurons=N, dimensions=1, label='B')

        nengo.Connection(input, A)
        nengo.Connection(A, B)
        nengo.Connection(B, A)

    # Put all objects on same partition
    partition = {input: 0, A: 0, B: 0}

    sim = nengo_mpi.Simulator(network, dt=0.001, fixed_nodes=partition)

    filename = 'nengo_serialization'

    pre_string = str(sim)

    sim.mpi_sim.write_to_file(filename)

    sim2 = nengo_mpi.Simulator(dt=0.001, fixed_nodes=partition)

    sim2.mpi_sim.read_from_file(filename)

    post_string = str(sim2)

    assert pre_string == post_string

if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])