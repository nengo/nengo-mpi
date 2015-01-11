from nengo.tests.conftest import *

import nengo
import nengo_mpi


def Mpi2Simulator(*args, **kwargs):
    return nengo_mpi.Simulator(*args, **kwargs)


def pytest_funcarg__Simulator(request):
    """the Simulator class being tested.

    For this file, it's sim_npy.Simulator.
    """
    return Mpi2Simulator


def pytest_funcarg__RefSimulator(request):
    """the Simulator class being tested.

    For this file, it's sim_npy.Simulator.
    """
    return nengo.Simulator
