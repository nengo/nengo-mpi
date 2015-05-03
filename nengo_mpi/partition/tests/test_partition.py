import pytest

from nengo_mpi import Simulator

from nengo_mpi import Partitioner
from nengo_mpi.partition import spectral_partitioner
from nengo_mpi.partition import work_balanced_partitioner
from nengo_mpi.partition import metis_available, metis_partitioner

from nengo_mpi.partition.base import network_to_filter_graph

import nengo

import os


@pytest.fixture
def simple_network():
    network = nengo.Network()

    with network:
        node = nengo.Node(0.5)
        A = nengo.Ensemble(50, 1, label='A')
        B = nengo.Ensemble(50, 1, label='B')

        nengo.Connection(node, A)
        nengo.Connection(A, B)

    return network


def pytest_generate_tests(metafunc):
    if "n_components" in metafunc.funcargnames:
        metafunc.parametrize("n_components", [1, 2, 4, 8])


def test_default(simple_network, n_components):

    # Save the network to a file so that an MpiSimulator is not created.
    save_file = 'test.net'

    component0, filter_graph = network_to_filter_graph(simple_network)

    partitioner = Partitioner(n_components)

    sim = Simulator(
        simple_network, partitioner=partitioner, save_file=save_file)

    assert sim.n_components == min(len(filter_graph), n_components)
    assert os.path.isfile(save_file)
    os.remove(save_file)


def test_spectral(simple_network):
    save_file = 'test.net'

    n_components = 2
    partitioner = Partitioner(n_components, func=spectral_partitioner)
    sim = Simulator(
        simple_network, partitioner=partitioner, save_file=save_file)

    assert sim.n_components == 2
    assert os.path.isfile(save_file)
    os.remove(save_file)


@pytest.mark.skipif(
    not metis_available(),
    reason="Requires metis to be available on the system.")
def test_metis(simple_network):
    save_file = 'test.net'

    n_components = 2
    partitioner = Partitioner(
        n_components, func=metis_partitioner, delete_file=True)
    sim = Simulator(
        simple_network, partitioner=partitioner, save_file=save_file)

    assert sim.n_components == n_components
    assert os.path.isfile(save_file)
    os.remove(save_file)


def test_work_balanced(simple_network):
    save_file = 'test.net'

    n_components = 2
    partitioner = Partitioner(n_components, func=work_balanced_partitioner)
    sim = Simulator(
        simple_network, partitioner=partitioner, save_file=save_file)

    assert sim.n_components == n_components
    assert os.path.isfile(save_file)
    os.remove(save_file)


def test_no_partitioner(simple_network):
    save_file = 'test.net'

    sim = Simulator(simple_network, save_file=save_file)

    assert sim.n_components == 1

    assert os.path.isfile(save_file)
    os.remove(save_file)


def test_assignments():
    save_file = 'test.net'

    network = nengo.Network()

    with network:
        node = nengo.Node(0.5)
        A = nengo.Ensemble(50, 1, label='A')
        B = nengo.Ensemble(50, 1, label='B')

        nengo.Connection(node, A)
        nengo.Connection(A, B)

    sim = Simulator(
        network, assignments={node: 0, A: 1, B: 2}, save_file=save_file)

    assert sim.n_components == 3

    assert os.path.isfile(save_file)
    os.remove(save_file)


def test_bad_assignments():
    save_file = 'test.net'

    network = nengo.Network()

    with network:
        node = nengo.Node(0.5)
        A = nengo.Ensemble(50, 1, label='A')
        B = nengo.Ensemble(50, 1, label='B')

        nengo.Connection(node, A)
        nengo.Connection(A, B, synapse=None)

    # No filter between A and B, but we're assigning them to different
    # partitions, so an error should be raised.
    with pytest.raises(RuntimeError):
        Simulator(
            network, assignments={node: 0, A: 1, B: 2}, save_file=save_file)


def test_insufficient_chunks(simple_network):
    """
    Test the case where the network contains too few chunks to use the
    specified number of components.
    """

    component0, filter_graph = network_to_filter_graph(simple_network)

    partitioner = Partitioner(len(filter_graph) + 10)

    save_file = 'test.net'
    sim = Simulator(
        simple_network, partitioner=partitioner, save_file=save_file)

    assert sim.n_components == len(filter_graph)
    assert os.path.isfile(save_file)
    os.remove(save_file)


def test_filter_graph():
    network = nengo.Network()

    with network:
        node = nengo.Node(0.5, label='Node 0')
        A = nengo.Ensemble(50, 1, label='A')
        B = nengo.Ensemble(50, 1, label='B')

        nengo.Connection(node, A)
        nengo.Connection(A, B)

    component0, filter_graph = network_to_filter_graph(network)
    assert len(filter_graph) == 2

    component0, filter_graph = network_to_filter_graph(
        network, merge_nengo_nodes=False)
    assert len(filter_graph) == 3

    with network:
        node = nengo.Node(lambda x: 0.5, label='Node 1')

    component0, filter_graph = network_to_filter_graph(network)
    assert len(filter_graph) == 2
    assert component0 is not None

    component0, filter_graph = network_to_filter_graph(
        network, merge_nengo_nodes=False)
    assert len(filter_graph) == 4
    assert component0 is not None

    with network:
        node = nengo.Node(lambda x: 0.7, label='Node 2')
        nengo.Connection(node, A, synapse=None)

    component0, filter_graph = network_to_filter_graph(network)
    assert len(filter_graph) == 2

    component0, filter_graph = network_to_filter_graph(
        network, merge_nengo_nodes=False)
    assert len(filter_graph) == 3

    with network:
        C = nengo.Ensemble(100, 1, label='C')
        nengo.Connection(C, A)

    component0, filter_graph = network_to_filter_graph(network)
    assert len(filter_graph) == 3

    component0, filter_graph = network_to_filter_graph(
        network, merge_nengo_nodes=False)
    assert len(filter_graph) == 4
