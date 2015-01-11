import numpy as np

import nengo
from nengo import spa

import nengo_mpi

seed = 1
model = spa.SPA(seed=seed)

with model:
    model.vision = spa.Buffer(dimensions=16)
    model.motor = spa.Buffer(dimensions=16)

    actions = spa.Actions(
        '0.5 --> motor=A',
        'dot(vision,CAT) --> motor=B',
        'dot(vision*CAT,DOG) --> motor=C',
        '2*dot(vision,CAT*0.5) --> motor=D',
        'dot(vision,CAT)+0.5-dot(vision,CAT) --> motor=E',
    )
    model.bg = spa.BasalGanglia(actions)

    def input(t):
        if t < 0.1:
            return '0'
        elif t < 0.2:
            return 'CAT'
        elif t < 0.3:
            return 'DOG*~CAT'
        else:
            return '0'
    model.input = spa.Input(vision=input)
    p = nengo.Probe(model.bg.input, 'output', synapse=0.03)

sim = nengo_mpi.Simulator(model)
sim.run(0.3)
t = sim.trange()

assert 0.6 > sim.data[p][t == 0.1, 0] > 0.4
assert sim.data[p][t == 0.2, 1] > 0.8
assert sim.data[p][-1, 2] > 0.6

assert np.allclose(sim.data[p][:, 1], sim.data[p][:, 3])
assert np.allclose(sim.data[p][:, 0], sim.data[p][:, 4])
