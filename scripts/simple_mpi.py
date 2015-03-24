import logging

import nengo

logger = logging.getLogger(__name__)
nengo.log(debug=False)

name = 'node_to_ensemble'
N = 300
D = 6

m = nengo.Network(label=name, seed=123)
with m:
    m.config[nengo.Ensemble].neuron_type = nengo.LIF()
    input_node = nengo.Node(output=[0.25] * D)
    a = nengo.Ensemble(N * D, dimensions=D)
    b = nengo.Ensemble(N * D, dimensions=D)
    c = nengo.Ensemble(N * D, dimensions=D)
    d = nengo.Ensemble(N * D, dimensions=D)
    e = nengo.Ensemble(N * D, dimensions=D)
    f = nengo.Ensemble(N * D, dimensions=D)
    g = nengo.Ensemble(N * D, dimensions=D)

    nengo.Connection(input_node, a)
    nengo.Connection(a, b)
    nengo.Connection(b, c, function=lambda x: x * 2)
    nengo.Connection(c, d, function=lambda x: x + 0.1)
    nengo.Connection(d, e)
    nengo.Connection(e, f)
    nengo.Connection(f, g)

    input_p = nengo.Probe(input_node, synapse=0.01)
    a_p = nengo.Probe(a, 'decoded_output', synapse=0.01)
    b_p = nengo.Probe(b, 'decoded_output', synapse=0.01)
    c_p = nengo.Probe(c, 'decoded_output', synapse=0.01)
    d_p = nengo.Probe(d, 'decoded_output', synapse=0.01)
    e_p = nengo.Probe(e, 'decoded_output', synapse=0.01)
    f_p = nengo.Probe(f, 'decoded_output', synapse=0.01)
    g_p = nengo.Probe(g, 'decoded_output', synapse=0.01)

sim_time = .1

if 1:
    import nengo_mpi
    partitioner = nengo_mpi.Partitioner(1)#, {a: 1, b: 2, c: 3, d: 1, e: 3, f: 2, g: 1})

    print "Building model..."
    sim = nengo_mpi.Simulator(m, dt=0.001, partitioner=partitioner)
    print "Done building model..."

    print "Running model for", sim_time, "seconds..."
    sim.run(sim_time)
    print "Done running model."
else:
    sim = nengo.Simulator(m, dt=0.001)
    sim.run(sim_time)


#t = sim.trange()
print sim.data[a_p][-10:]
print sim.data[f_p][-10:]
print sim.data[g_p][-10:]
