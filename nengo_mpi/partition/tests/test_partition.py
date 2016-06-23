import pytest
import os
import numpy as np

import nengo

from nengo_mpi import Simulator
from nengo_mpi import Partitioner, PartitionError
from nengo_mpi.partition import work_balanced_partitioner
from nengo_mpi.partition import metis_available, metis_partitioner
from nengo_mpi.partition.base import network_to_cluster_graph, make_boundary_predicate


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

    bp = make_boundary_predicate(simple_network)
    component0, cluster_graph = network_to_cluster_graph(simple_network, bp)

    partitioner = Partitioner(n_components)

    sim = Simulator(
        simple_network, partitioner=partitioner, save_file=save_file)

    try:
        assert sim.n_components == min(len(cluster_graph), n_components)
        assert os.path.isfile(save_file)
    finally:
        try:
            os.remove(save_file)
        except:
            pass


@pytest.mark.skipif(
    not metis_available(),
    reason="Requires metis to be available on the system.")
def test_metis(simple_network):
    save_file = 'test.net'

    n_components = 2
    partitioner = Partitioner(
        n_components, func=metis_partitioner, delete_file=True)
    try:
        sim = Simulator(
            simple_network, partitioner=partitioner, save_file=save_file)

        assert sim.n_components == n_components
        assert os.path.isfile(save_file)
    finally:
        try:
            os.remove(save_file)
        except:
            pass


def test_work_balanced(simple_network):
    save_file = 'test.net'

    n_components = 2
    partitioner = Partitioner(n_components, func=work_balanced_partitioner)
    sim = Simulator(
        simple_network, partitioner=partitioner, save_file=save_file)

    try:
        assert sim.n_components == n_components
        assert os.path.isfile(save_file)
    finally:
        try:
            os.remove(save_file)
        except:
            pass


def test_no_partitioner(simple_network):
    save_file = 'test.net'

    try:
        sim = Simulator(simple_network, save_file=save_file)
        assert sim.n_components == 1
        assert os.path.isfile(save_file)
    finally:
        try:
            os.remove(save_file)
        except:
            pass


def test_assignments():
    save_file = 'test.net'

    network = nengo.Network()

    with network:
        node = nengo.Node(0.5)
        A = nengo.Ensemble(50, 1, label='A')
        B = nengo.Ensemble(50, 1, label='B')

        nengo.Connection(node, A)
        nengo.Connection(A, B)

    try:
        sim = Simulator(
            network, assignments={node: 0, A: 1, B: 2}, save_file=save_file)
        assert sim.n_components == 3
        assert os.path.isfile(save_file)
    finally:
        try:
            os.remove(save_file)
        except:
            pass


def test_bad_assignments():
    save_file = 'test.net'

    network = nengo.Network()

    with network:
        node = nengo.Node(0.5)
        A = nengo.Ensemble(50, 1, label='A')
        B = nengo.Ensemble(50, 1, label='B')

        nengo.Connection(node, A)
        nengo.Connection(A, B, synapse=None)

    try:
        # No synapse between A and B, but we're assigning them to different
        # partitions, so an error should be raised.
        with pytest.raises(PartitionError):
            Simulator(
                network, assignments={node: 0, A: 1, B: 2},
                save_file=save_file)
    finally:
        try:
            os.remove(save_file)
        except:
            pass


def test_insufficient_chunks(simple_network):
    """
    Test the case where the network contains too few chunks to use the
    specified number of components.
    """

    bp = make_boundary_predicate(simple_network)
    component0, cluster_graph = network_to_cluster_graph(simple_network, bp)

    partitioner = Partitioner(len(cluster_graph) + 10)

    save_file = 'test.net'

    try:
        sim = Simulator(
            simple_network, partitioner=partitioner, save_file=save_file)
        assert sim.n_components == len(cluster_graph)
        assert os.path.isfile(save_file)
    finally:
        try:
            os.remove(save_file)
        except:
            pass


def test_cluster_graph():
    network = nengo.Network()

    with network:
        node = nengo.Node(0.5, label='Node 0')
        A = nengo.Ensemble(50, 1, label='A')
        B = nengo.Ensemble(50, 1, label='B')

        nengo.Connection(node, A)
        nengo.Connection(A, B)

    bp = make_boundary_predicate(network)

    component0, cluster_graph = network_to_cluster_graph(network, bp)
    assert len(cluster_graph) == 2

    component0, cluster_graph = network_to_cluster_graph(
        network, bp, merge_nengo_nodes=False)
    assert len(cluster_graph) == 3

    with network:
        node = nengo.Node(lambda x: 0.5, label='Node 1')

    component0, cluster_graph = network_to_cluster_graph(network, bp)
    assert len(cluster_graph) == 2
    assert component0 is not None

    component0, cluster_graph = network_to_cluster_graph(
        network, bp, merge_nengo_nodes=False)
    assert len(cluster_graph) == 4
    assert component0 is not None

    with network:
        node = nengo.Node(lambda x: 0.7, label='Node 2')
        nengo.Connection(node, A, synapse=None)

    component0, cluster_graph = network_to_cluster_graph(network, bp)
    assert len(cluster_graph) == 2

    component0, cluster_graph = network_to_cluster_graph(
        network, bp, merge_nengo_nodes=False)
    assert len(cluster_graph) == 3

    with network:
        C = nengo.Ensemble(100, 1, label='C')
        nengo.Connection(C, A)

    component0, cluster_graph = network_to_cluster_graph(network, bp)
    assert len(cluster_graph) == 3

    component0, cluster_graph = network_to_cluster_graph(
        network, bp, merge_nengo_nodes=False)
    assert len(cluster_graph) == 4


def test_learning_rules(rng):
    """ Test that connections with learning rules are not allowed
        to cross boundaries. """

    save_file = 'test.net'
    network = nengo.Network()

    with network:
        node = nengo.Node(0.5, label='Node 0')
        A = nengo.Ensemble(50, 1, label='A')
        B = nengo.Ensemble(50, 1, label='B')

        nengo.Connection(node, A)

        initial_weights = rng.uniform(
            high=1e-3, size=(A.n_neurons, B.n_neurons))
        nengo.Connection(
            A.neurons, B.neurons,
            transform=initial_weights,
            learning_rule_type=nengo.learning_rules.Oja())

    try:
        with pytest.raises(PartitionError):
            Simulator(
                network, assignments={node: 0, A: 1, B: 2},
                save_file=save_file)
    finally:
        try:
            os.remove(save_file)
        except:
            pass


def test_probed_connection():
    """ Test that probed connections are not allowed to cross boundaries. """

    save_file = 'test.net'
    network = nengo.Network()

    with network:
        node = nengo.Node(0.5, label='Node 0')
        A = nengo.Ensemble(50, 1, label='A')
        B = nengo.Ensemble(50, 1, label='B')

        nengo.Connection(node, A)

        C = nengo.Connection(A, B)
        nengo.Probe(C)

    try:
        with pytest.raises(PartitionError):
            Simulator(
                network, assignments={node: 0, A: 1, B: 2},
                save_file=save_file)

    finally:
        try:
            os.remove(save_file)
        except:
            pass


@pytest.mark.parametrize('do_partition', [True, False])
def test_pes_partition(
        do_partition, seed, pre_neurons=False, post_neurons=False,
        weight_solver=False, vin=np.array([0.5, -0.5]), vout=None,
        n=200, function=None, transform=np.array(1.), rate=1e-3):

    vout = np.array(vin) if vout is None else vout
    model = nengo.Network(seed=seed)
    with model:
        u = nengo.Node(output=vin)
        v = nengo.Node(output=vout)
        a = nengo.Ensemble(n, dimensions=u.size_out)
        b = nengo.Ensemble(n, dimensions=u.size_out)
        e = nengo.Ensemble(n, dimensions=v.size_out)

        nengo.Connection(u, a)

        bslice = b[:v.size_out] if v.size_out < u.size_out else b
        pre = a.neurons if pre_neurons else a
        post = b.neurons if post_neurons else bslice

        conn = nengo.Connection(pre, post,
                                function=function, transform=transform,
                                learning_rule_type=nengo.PES(rate))
        if weight_solver:
            conn.solver = nengo.solvers.LstsqL2(weights=True)

        nengo.Connection(v, e, transform=-1)
        nengo.Connection(bslice, e)
        nengo.Connection(e, conn.learning_rule)

        nengo.Probe(bslice, synapse=0.03)
        nengo.Probe(e, synapse=0.03)

        nengo.Probe(conn, 'weights', sample_every=0.01)
        nengo.Probe(conn.learning_rule, 'correction', synapse=0.03)

    save_file = 'test.net'

    try:
        if do_partition:
            partitioner = Partitioner(2)
            Simulator(
                model, partitioner=partitioner, save_file=save_file)
        else:
            assignments = {u: 0, v: 1, a: 0, b: 0, e: 1}
            Simulator(model, assignments=assignments, save_file=save_file)
    finally:
        try:
            os.remove(save_file)
        except:
            pass
