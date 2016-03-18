'''
Black-box testing of the sim_ocl Simulator.

TestCase classes are added automatically from
nengo.tests, but you can still run individual
test files like this:

$ py.test test/test_sim_ocl.py -k test_ensemble.test_scalar

See http://pytest.org/latest/usage.html for more invocations.

'''
import fnmatch
import os
import sys
import pytest

import nengo
from nengo.utils.testing import find_modules

from nengo_mpi.tests.utils import load_functions


def xfail(pattern, msg):
    for key in tests:
        if fnmatch.fnmatch(key, pattern):
            tests[key] = pytest.mark.xfail(True, reason=msg)(tests[key])


nengo_dir = os.path.dirname(nengo.__file__)
modules = find_modules(nengo_dir, prefix='nengo')
tests = load_functions(modules, arg_pattern='^(Ref)?Simulator$')

# noise
xfail('test.nengo.tests.test_ensemble.test_noise_copies_ok',
      'nengo_mpi does not support FilteredNoise')
xfail('test.nengo.tests.test_processes.test_brownnoise',
      'nengo_mpi does not support FilteredNoise')

# nodes
xfail('test.nengo.tests.test_node.test_none',
      'No error if nodes output None')
xfail('test.nengo.tests.test_node.test_unconnected_node',
      'Unconnected nodes not supported')
xfail('test.nengo.tests.test_node.test_set_output',
      'Unconnected nodes not supported')
xfail('test.nengo.tests.test_node.test_args',
      'This test fails for an unknown reason')

# izhikevich neurons not implemented.
xfail('test.nengo.tests.test_neurons.test_izhikevich',
      'This test fails for an unknown reason')

# cache
xfail('test.nengo.tests.test_cache.test_cache_works',
      'Not set up correctly.')

# opens multiple simulators
xfail('test.nengo.tests.test_probe.test_dts',
      'Opens multiple simulators')
xfail('test.nengo.tests.test_connection.test_shortfilter',
      'Opens multiple simulators')
xfail('test.nengo.tests.test_connection.test_set_function',
      'Opens multiple simulators')
xfail('test.nengo.tests.test_neurons.test_dt_dependence',
      'Opens multiple simulators')
xfail('test.nengo.tests.test_learning_rules.test_dt_dependence',
      'Opens multiple simulators')

locals().update(tests)


if __name__ == '__main__':
    pytest.main(sys.argv)
