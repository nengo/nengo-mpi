"""
Black-box testing of the MPI Simulator.

Modified from the same file in the nengo-ocl.

TestCase classes are added automatically from
nengo.tests, but you can still run individual
test files like this:

$ py.test test/test_simulator.py -k test_ensemble.test_scalar

See http://pytest.org/latest/usage.html for more invocations.

"""

import inspect
import os.path

import nengo
import pytest

test_files = {}

if 0:
    # Standard nengo tests
    nengo_test_dir = os.path.dirname(nengo.tests.__file__)
    test_files["nengo.tests."] = os.listdir(nengo_test_dir)

try:
    if 1:
        # Spa tests
        import nengo.spa.tests
        spa_test_dir = os.path.dirname(nengo.spa.tests.__file__)
        test_files["nengo.spa.tests."] = os.listdir(spa_test_dir)

except Exception as e:
    print "Couldn't import spa tests because" + e.message

try:
    if 0:
        # Nengo network tests
        import nengo.networks.tests
        networks_test_dir = os.path.dirname(nengo.networks.tests.__file__)
        test_files["nengo.networks.tests."] = os.listdir(networks_test_dir)

except Exception as e:
    print "Couldn't import network tests because" + e.message

nengo.log(debug=False)

for key in test_files:
    for test_file in test_files[key]:
        if not test_file.startswith('test_') or not test_file.endswith('.py'):
            continue

        if test_file.startswith('test_examples'):
            continue

        module_string = key + test_file[:-3]
        m = __import__(module_string, globals(), locals(), ['*'])

        for k in dir(m):
            if k.startswith('test_'):
                tst = getattr(m, k)
                if callable(tst):
                    args = inspect.getargspec(tst).args
                    if 'Simulator' in args:
                        locals()[test_file[:-3] + '.' + k] = tst
            if k.startswith('pytest'):
                locals()[k] = getattr(m, k)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
