import pytest
from nengo.conftest import *

import nengo_mpi


@pytest.fixture(scope='function')
def Simulator(request):
    '''The Simulator class being tested.

    This hides the Simulator fixture provided by nengo.conftest

    '''
    request.addfinalizer(nengo_mpi.Simulator.close_simulators)
    return nengo_mpi.Simulator

TestConfig.Simulator = nengo_mpi.Simulator
