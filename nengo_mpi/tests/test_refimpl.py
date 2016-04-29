'''
Black-box testing of the sim_ocl Simulator.

TestCase classes are added automatically from
nengo.tests, but you can still run individual
test files like this:

$ py.test test/test_sim_ocl.py -k test_ensemble.test_scalar

See http://pytest.org/latest/usage.html for more invocations.

'''
import os
import sys
import pytest

import nengo
from nengo.utils.testing import find_modules
from nengo_mpi.tests.utils import load_functions

nengo_dir = os.path.dirname(nengo.__file__)
modules = find_modules(nengo_dir, prefix='nengo')
tests = load_functions(modules, arg_pattern='^Simulator$')

locals().update(tests)


if __name__ == '__main__':
    pytest.main(sys.argv)
