from nengo.tests.conftest import *

import nengo_mpi
import nengo


@pytest.fixture(scope='function')
def Simulator(request):
    '''The Simulator class being tested.'''

    request.addfinalizer(nengo_mpi.Simulator.close_simulators)
    return nengo_mpi.Simulator


def pytest_funcarg__RefSimulator(request):
    '''the Simulator class being tested.'''
    return nengo.Simulator


def pytest_addoption(parser):
    parser.addoption(
        '--plots', nargs='?', default=False, const=True,
        help='Save plots (can optionally specify a directory for plots).')
    parser.addoption(
        '--analytics', nargs='?', default=False, const=True,
        help='Save analytics (can optionally specify a directory for data).')
    parser.addoption(
        '--compare', nargs=2, default=None,
        help='Compare analytics results (specify directories to compare).')
    parser.addoption(
        '--logs', nargs='?', default=False, const=True,
        help='Save logs (can optionally specify a directory for logs).')
    parser.addoption('--noexamples', action='store_false', default=True,
                     help='Do not run examples')
    parser.addoption(
        '--slow', action='store_true', default=False,
        help='Also run slow tests.')
