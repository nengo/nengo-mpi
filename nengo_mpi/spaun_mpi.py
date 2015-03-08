# from nengo.base import NengoObject
from nengo.builder.signal import Signal
from nengo.builder.operator import Operator
from nengo.node import Node

import numpy as np


class SpaunStimulusOperator(Operator):
    def __init__(self, output, stimulus_sequence):

        self.output = output
        self.stimulus_sequence = stimulus_sequence

        self.sets = []
        self.incs = []
        self.reads = []
        self.updates = []

    def make_step(self, signals, dt, rng):
        def step():
            pass

        return step


class SpaunStimulus(Node):
    def __init__(
            self, dimension, stimulus_sequence, label=None):

        super(SpaunStimulus, self).__init__(
            output=np.zeros(dimension), label=label)

        self.dimension = dimension
        self.stimulus_sequence = stimulus_sequence


def build_spaun_stimulus(model, ss):
    print "In build spaun stimulus..."

    output = Signal(np.zeros(ss.dimension), name=str(ss))

    # Allows build_connection to get the output signal
    model.sig[ss]['out'] = output

    model.add_op(SpaunStimulusOperator(output, ss.stimulus_sequence))

