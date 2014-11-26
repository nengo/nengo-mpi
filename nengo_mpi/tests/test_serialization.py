import logging
import pytest

import nengo
import nengo_mpi

logger = logging.getLogger(__name__)


@pytest.mark.xfail
def test_serialization():
    """Test the serialization of MPISimulator at the C++ level."""

    N = 30
    val = 0.5

    network = nengo.Network(label='test_serialization', seed=123)
    with network:
        input = nengo.Node(output=val, label='input')
        A = nengo.Ensemble(n_neurons=N, dimensions=1, label='A')
        B = nengo.Ensemble(n_neurons=N, dimensions=1, label='B')

        nengo.Connection(input, A)
        nengo.Connection(A, B)
        nengo.Connection(B, A)

    sim = nengo_mpi.Simulator(network, dt=0.001)

    filename = 'nengo_serialization'

    pre_string = str(sim)

    sim.mpi_sim.write_to_file(filename)

    print "pre_string:", pre_string
    sim2 = nengo_mpi.Simulator(nengo.Network(), dt=0.001)

    sim2.mpi_sim.read_from_file(filename)

    post_string = str(sim2)

    print "post_string:", post_string

    assert pre_string == post_string

if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
