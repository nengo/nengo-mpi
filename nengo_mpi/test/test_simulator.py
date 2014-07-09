"""
Black-box testing of the MPI Simulator.

TestCase classes are added automatically from
nengo.tests, but you can still run individual
test files like this:

$ py.test test/test_simulator.py -k test_ensemble.test_scalar

See http://pytest.org/latest/usage.html for more invocations.

"""

import inspect
import os.path

import nengo.tests
import pytest

import nengo_mpi

nengotestdir = os.path.dirname(nengo.tests.__file__)
nengo.log(debug=True)

for testfile in os.listdir(nengotestdir):
    if not testfile.startswith('test_') or not testfile.endswith('.py'):
        continue

    if testfile.startswith('test_examples'):
        continue

    module_string = "nengo.tests." + testfile[:-3]
    m = __import__(module_string, globals(), locals(), ['*'])
    for k in dir(m):
        if k.startswith('test_'):
            tst = getattr(m, k)
            args = inspect.getargspec(tst).args
            if 'Simulator' in args:
                locals()[testfile[:-3] + '.' + k] = tst
        if k.startswith('pytest'):
            locals()[k] = getattr(m, k)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
