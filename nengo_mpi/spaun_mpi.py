from nengo.builder.signal import Signal
from nengo.builder.operator import Operator
from nengo.node import Node

import numpy as np


class SpaunStimulusOperator(Operator):
    """
    A placeholder operator meant to store the parameters of the Spaun
    Stimulus, and forward them to the C++ code.
    """

    def __init__(
            self, output, stimulus_sequence, present_interval, present_blanks):

        self.output = output
        self.stimulus_sequence = stimulus_sequence
        self.present_interval = present_interval
        self.present_blanks = present_blanks

        self.sets = []
        self.incs = []
        self.reads = []
        self.updates = []

    def make_step(self, signals, dt, rng):
        def step():
            pass

        return step


class SpaunStimulus(Node):
    """
    A placeholder nengo object.
    When built, creates an instance of SpaunStimulusOperator.
    """

    def __init__(
            self, dimension, stimulus_sequence,
            present_interval, present_blanks, label=None):

        super(SpaunStimulus, self).__init__(
            output=np.zeros(dimension), label=label)

        self.dimension = dimension
        self.stimulus_sequence = [
            "NULL" if not s else s for s in stimulus_sequence]

        self.present_interval = float(present_interval)
        self.present_blanks = float(present_blanks)


def build_spaun_stimulus(model, ss):
    output = Signal(np.zeros(ss.dimension), name=str(ss))

    # Allows build_connection to get the output signal
    model.sig[ss]['out'] = output

    op = SpaunStimulusOperator(
        output, ss.stimulus_sequence, ss.present_interval, ss.present_blanks)

    model.add_op(op)
