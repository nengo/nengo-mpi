import logging

import numpy as np

import nengo
import nengo_mpi
from nengo.builder import ShapeMismatch
from nengo.utils.numpy import rmse, norm

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

#def test_product(Simulator, nl):
#    N = 80
#
#    m = nengo.Network(label='test_product', seed=124)
#    with m:
#        sin = nengo.Node(output=np.sin)
#        cons = nengo.Node(output=-.5)
#        factors = nengo.Ensemble(nl(2 * N), dimensions=2, radius=1.5)
#        if nl != nengo.Direct:
#            factors.encoders = np.tile(
#                [[1, 1], [-1, 1], [1, -1], [-1, -1]],
#                (factors.n_neurons // 4, 1))
#        product = nengo.Ensemble(nl(N), dimensions=1)
#        nengo.Connection(sin, factors[0])
#        nengo.Connection(cons, factors[1])
#        nengo.Connection(
#            factors, product, function=lambda x: x[0] * x[1], filter=0.01)
#
#        sin_p = nengo.Probe(sin, 'output', sample_every=.01)
#        # TODO
#        # m.probe(conn, sample_every=.01)
#        factors_p = nengo.Probe(
#            factors, 'decoded_output', sample_every=.01, filter=.01)
#        product_p = nengo.Probe(
#            product, 'decoded_output', sample_every=.01, filter=.01)
#
#    sim = Simulator(m)
#    sim.run(6)
#
#    t = sim.trange(dt=.01)
#    plt.subplot(211)
#    plt.plot(t, sim.data[factors_p][:, 0], label='factors[0]')
#    plt.plot(t, sim.data[factors_p][:, 1], label='factors[1]')
#    plt.plot(t, np.sin(np.arange(0, 6, .01)), label='ideal sin')
#    plt.plot(t, sim.data[sin_p], label='sin')
#    plt.legend(loc='best')
#    plt.subplot(212)
#    plt.plot(t, sim.data[product_p], label='product')
#    # TODO
#    # plt.plot(sim.data[conn])
#    plt.plot(t, -.5 * np.sin(np.arange(0, 6, .01)), label='ideal product')
#    plt.legend(loc='best')
#    plt.savefig('fig.pdf')
#    plt.close()
#
#    sin = np.sin(np.arange(0, 6, .01))
#    assert rmse(sim.data[factors_p][:, 0], sin) < 0.1
#    assert rmse(sim.data[factors_p][20:, 1], -0.5) < 0.1
#
#    assert rmse(sim.data[product_p][:, 0], -0.5 * sin) < 0.1
#    # assert rmse(sim.data[conn][:, 0], -0.5 * sin) < 0.1

def test_constant_scalar(Simulator, nl):
    """A Network that represents a constant value."""
    N = 30
    val = 0.5

    m = nengo.Network(label='test_constant_scalar', seed=123)
    with m:
        input = nengo.Node(output=val, label='input')
        A = nengo.Ensemble(n_neurons=N, dimensions=1, neuron_type=nl())
        nengo.Connection(input, A)
        in_p = nengo.Probe(input, 'output')
        A_p = nengo.Probe(A, 'decoded_output', synapse=0.1) 
        #spike_probe = nengo.Probe(A, 'spikes')

    sim = Simulator(m, dt=0.001)
    sim.run(1.0)

    t = sim.trange()
    plt.plot(t, sim.data[in_p], label='Input')
    plt.plot(t, sim.data[A_p], label='Neuron approximation, pstc=0.1')
    plt.ylim((-0.1, val+.1))
    plt.legend(loc=0)
    plt.savefig('test_ensemble.test_constant_scalar.pdf')
    plt.close()

    assert np.allclose(sim.data[in_p].ravel(), val, atol=.1, rtol=.01)
    assert np.allclose(sim.data[A_p][-10:], val, atol=.1, rtol=.01)

if __name__ == "__main__":
    nengo.log(debug=True)
    test_constant_scalar(nengo_mpi.Simulator, nengo.LIFRate)

    #test_product(nengo_mpi.Simulator, nengo.LIF)
    #test_product(nengo_mpi.Simulator, nengo.LIFRate)

