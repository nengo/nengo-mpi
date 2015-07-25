"""
Black-box testing of the sim_ocl Simulator.

TestCase classes are added automatically from
nengo.tests, but you can still run individual
test files like this:

$ py.test test/test_sim_ocl.py -k test_ensemble.test_scalar

See http://pytest.org/latest/usage.html for more invocations.

"""
import fnmatch
import os
import sys
import pytest

import nengo
from nengo.utils.testing import find_modules

import nengo_mpi
from nengo_mpi.tests.utils import load_functions


def xfail(pattern, msg):
    for key in tests:
        if fnmatch.fnmatch(key, pattern):
            tests[key] = pytest.mark.xfail(True, reason=msg)(tests[key])


def MpiSimulator(*args, **kwargs):
    return nengo_mpi.Simulator(*args, **kwargs)


def pytest_funcarg__Simulator(request):
    """the Simulator class being tested."""
    return MpiSimulator


def pytest_funcarg__RefSimulator(request):
    """the Simulator class being tested."""
    return nengo.Simulator


nengo_dir = os.path.dirname(nengo.__file__)
modules = find_modules(nengo_dir, prefix='nengo')
tests = load_functions(modules, arg_pattern='^(Ref)?Simulator$')

# noise
xfail('test.nengo.tests.test_ensemble.test_noise*',
      "nengo_mpi does not support noise")
xfail('test.nengo.tests.test_simulator.test_noise*',
      "nengo_mpi does not support noise")
xfail("test.nengo.utils.tests.test_neurons.test_rates_kernel",
      'Uses noise.')
xfail('test.nengo.utils.tests.test_neurons.test_rates_isi',
      'Uses noise')
xfail('test.nengo.tests.test_synapses.test_linearfilter',
      'Uses noise')
xfail('test.nengo.tests.test_synapses.test_lowpass',
      'Uses noise')
xfail('test.nengo.tests.test_synapses.test_decoders',
      'Uses noise')
xfail('test.nengo.tests.test_neurons.test_dt_dependence*',
      'Uses noise')
xfail('test.nengo.tests.test_probe.test_defaults',
      'Uses noise')
xfail('test.nengo.tests.test_synapses.test_triangle',
      'Uses noise')
xfail('test.nengo.tests.test_neurons.test_izhikevich',
      'Uses noise')

# learning rules
# xfail('test.nengo.tests.test_learning_rules.test_unsupervised',
#       "Unsupervised learning rules not implemented")
# xfail('test.nengo.tests.test_learning_rules.test_dt_dependence',
#       "Filtering matrices (i.e. learned transform) not implemented")
xfail('test.nengo.tests.test_learning_rules.*',
      "Learning rules do not currently work in nengo_mpi."
      'Waiting for 2.1 release of nengo.')

# nodes
xfail('test.nengo.tests.test_node.test_none',
      "No error if nodes output None")
xfail('test.nengo.tests.test_node.test_unconnected_node',
      "Unconnected nodes not supported")
xfail('test.nengo.tests.test_node.test_set_output',
      "Unconnected nodes not supported")
xfail('test.nengo.tests.test_node.test_args',
      "This test fails for an unknown reason")

# synapses
xfail('test.nengo.tests.test_synapses.test_alpha',
      "Only first-order filters implemented")
xfail('test.nengo.tests.test_synapses.test_general',
      "Only first-order filters implemented")

# resetting
xfail('test.nengo.tests.test_learning_rules.test_reset',
      "Resetting not implemented")
xfail('test.nengo.tests.test_neurons.test_reset',
      "Resetting not implemented")

# cache
xfail('test.nengo.tests.test_cache.test_cache_works',
      'Not set up correctly.')

# connection probes
xfail('test.nengo.tests.test_connection.test_decoder_probe',
      'Cannot probe connections in nengo_mpi')
xfail('test.nengo.tests.test_connection.test_transform_probe',
      'Cannot probe connections in nengo_mpi')

locals().update(tests)


if __name__ == '__main__':
    pytest.main(sys.argv)
