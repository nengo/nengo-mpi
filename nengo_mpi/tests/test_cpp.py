import os
import subprocess
import pytest
import h5py

import numpy as np

import nengo_mpi
import nengo
from nengo.neurons import LIF, LIFRate, RectifiedLinear, Sigmoid
from nengo.neurons import AdaptiveLIF, AdaptiveLIFRate, Izhikevich

all_neurons = [
    LIF, LIFRate, RectifiedLinear, Sigmoid,
    AdaptiveLIF, AdaptiveLIFRate]  # Izhikevich]


@pytest.mark.parametrize("neuron_type", all_neurons)
@pytest.mark.parametrize("synapse", [None, 0.0, 0.02, 0.05])
def test_basic_cpp(neuron_type, synapse):
    n_neurons = 40

    m = nengo.Network(seed=1)
    with m:
        A = nengo.Ensemble(
            n_neurons, dimensions=3, neuron_type=neuron_type())

        B = nengo.Ensemble(
            n_neurons, dimensions=3, neuron_type=neuron_type())

        nengo.Connection(A, B, synapse=synapse)

        A_p = nengo.Probe(A)
        B_p = nengo.Probe(B)

        input = nengo.Node([0.1, 0.2, -0.3])
        nengo.Connection(input, A, synapse=0.05)

    sim_time = 0.01

    refimpl_sim = nengo.Simulator(m)
    refimpl_sim.run(sim_time)

    network_file = "test_nengo.net"
    log_file = "test_nengo.h5"

    try:
        nengo_mpi.Simulator(m, save_file=network_file)
        subprocess.check_output(
            ['nengo_cpp', '--noprog', network_file, str(sim_time)])

        results = h5py.File(log_file, 'r')
    finally:
        try:
            os.remove(network_file)
        except:
            pass

        try:
            os.remove(log_file)
        except:
            pass

    assert np.allclose(
        refimpl_sim.data[A_p], results[str(id(A_p))], atol=0.00001, rtol=0.00)
    assert np.allclose(
        refimpl_sim.data[B_p], results[str(id(B_p))], atol=0.00001, rtol=0.00)
