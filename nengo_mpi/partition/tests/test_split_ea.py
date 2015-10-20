import numpy as np
import os

import nengo
from nengo.networks.circularconvolution import circconv
from nengo.utils.numpy import rmse

from nengo_mpi.partition import EnsembleArraySplitter


def remove_log_file(splitter):
    assert os.path.isfile(splitter.log_file_name)
    os.remove(splitter.log_file_name)


def test_simple_split():
    model = nengo.Network()

    dims = 4
    seed = 1
    npd = 200
    sim_time = 1.0

    rng = np.random.RandomState(seed)
    a = rng.normal(size=dims)
    a /= np.linalg.norm(a)
    model.config[nengo.Ensemble].neuron_type = nengo.LIFRate()

    with model:
        input = nengo.Node(a)
        A = nengo.networks.EnsembleArray(npd, dims, radius=np.sqrt(1./dims))
        B = nengo.networks.EnsembleArray(npd, dims, radius=np.sqrt(1./dims))

        nengo.Connection(input, A.input)
        nengo.Connection(A.output, B.input)

        p = nengo.Probe(B.output)

    sim_no_split = nengo.Simulator(model)
    sim_no_split.run(sim_time)

    splitter = EnsembleArraySplitter()

    splitter.split(model, max_neurons=npd, preserve_zero_conns=False)

    assert len(model.networks) == 2
    assert len(model.all_networks) == 2
    assert len(model.all_ensembles) == 2 * dims
    assert len(model.all_ensembles) == 2 * dims
    assert len(model.ensembles) == 0
    assert len(model.all_nodes) == 4 * dims + 1
    assert len(model.all_connections) == 6 * dims

    sim_split = nengo.Simulator(model)
    sim_split.run(sim_time)

    pre_split_data = sim_no_split.data
    post_split_data = splitter.unsplit_data(sim_split)

    assert np.allclose(
        pre_split_data[p][-10:], post_split_data[p][-10:], atol=0.1)

    remove_log_file(splitter)


def test_circconv_split():

    dims = 16
    seed = 1
    npd = 100
    sim_time = 0.1

    rng = np.random.RandomState(seed)

    magnitude = 1.0
    pstc = 0.005

    a = rng.normal(scale=np.sqrt(1./dims), size=dims) * magnitude
    b = rng.normal(scale=np.sqrt(1./dims), size=dims) * magnitude
    result = circconv(a, b)

    model = nengo.Network(label="CircConv", seed=seed)
    model.config[nengo.Ensemble].neuron_type = nengo.LIFRate()

    with model:
        inputA = nengo.Node(a)
        inputB = nengo.Node(b)

        input_ea_a = nengo.networks.EnsembleArray(
            npd, dims, radius=np.sqrt(1./dims), label="A")
        input_ea_b = nengo.networks.EnsembleArray(
            npd, dims, radius=np.sqrt(1./dims), label="B")

        nengo.Connection(inputA, input_ea_a.input, synapse=None)
        nengo.Connection(inputB, input_ea_b.input, synapse=None)

        cconv = nengo.networks.CircularConvolution(
            npd, dimensions=dims,
            input_magnitude=magnitude)

        nengo.Connection(input_ea_a.output, cconv.A, synapse=pstc)
        nengo.Connection(input_ea_b.output, cconv.B, synapse=pstc)

        output = nengo.networks.EnsembleArray(
            npd, dims, radius=np.sqrt(1./dims), label="output")

        nengo.Connection(cconv.output, output.input, synapse=pstc)

        p = nengo.Probe(output.output)

    sim_no_split = nengo.Simulator(model)
    sim_no_split.run(sim_time)

    splitter = EnsembleArraySplitter()

    splitter.split(model, max_neurons=npd, preserve_zero_conns=False)

    assert len(model.networks) == 4
    assert len(model.all_networks) == 7
    assert len(model.ensembles) == 0
    assert len(model.all_ensembles) == 3 * dims + 2 * (2 * dims + 4)

    sim_split = nengo.Simulator(model)
    sim_split.run(sim_time)

    pre_split_data = sim_no_split.data
    post_split_data = splitter.unsplit_data(sim_split)

    error = rmse(result, pre_split_data[p][-1])
    assert error < 0.1

    error = rmse(result, post_split_data[p][-1])
    assert error < 0.1

    error = rmse(pre_split_data[p][-1], post_split_data[p][-1])
    assert error < 0.1

    remove_log_file(splitter)

if __name__ == "__main__":
    test_circconv_split()
