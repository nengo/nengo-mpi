import six
import numpy as np
import ctypes
from ctypes import (
    byref, POINTER, CFUNCTYPE, c_int, c_double, c_float,
    c_ulonglong, c_size_t, c_char_p)

from nengo.exceptions import SimulationError

from nengo_mpi.utils import signal_to_string

_native_sim_available = False
try:
    mpi_sim_so = ctypes.CDLL("mpi_sim.so")

    mpi_sim_so.create_simulator.argtypes = []
    mpi_sim_so.create_simulator.restype = None

    mpi_sim_so.load_network.argtypes = [c_char_p]
    mpi_sim_so.load_network.restype = None

    mpi_sim_so.finalize_build.argtypes = []
    mpi_sim_so.finalize_build.restype = None

    mpi_sim_so.run_n_steps.argtypes = [c_int, c_int, c_char_p]
    mpi_sim_so.run_n_steps.restype = None

    mpi_sim_so.get_probe_data.argtypes = [
        c_ulonglong, POINTER(c_size_t), POINTER(c_size_t)]
    mpi_sim_so.get_probe_data.restype = POINTER(c_double)

    mpi_sim_so.get_signal_value.argtypes = [
        c_ulonglong, POINTER(c_size_t), POINTER(c_size_t)]
    mpi_sim_so.get_signal_value.restype = POINTER(c_double)

    mpi_sim_so.free_ptr.argtypes = [POINTER(c_double)]
    mpi_sim_so.free_ptr.restype = None

    mpi_sim_so.reset_simulator.argtypes = [c_ulonglong]
    mpi_sim_so.reset_simulator.restype = None

    mpi_sim_so.close_simulator.argtypes = []
    mpi_sim_so.close_simulator.restype = None

    SIMPLE_PYFUNC = CFUNCTYPE(None)
    mpi_sim_so.create_PyFunc.argtypes = [
        SIMPLE_PYFUNC, c_char_p, c_char_p, c_char_p,
        POINTER(c_double), POINTER(c_double), POINTER(c_double)]
    mpi_sim_so.create_PyFunc.restype = None

    _native_sim_available = True
except ImportError as e:
    print("mpi_sim.so not available. Reason:\n" +
          ("    %s.\n" % e) +
          "Network files may be created, but simulations cannot be run.")
    _native_sim_available = False


def native_sim_available():
    return _native_sim_available


class NativeSimulator(object):
    """ A python wrapper for the native simulator implemented by mpi_sim.so.

    Talks to the native simulator using ctypes.

    """
    def __init__(self, sig):
        if not native_sim_available():
            raise Exception(
                "Created NativeSimulator, but mpi_sim.so is not available.")

        self.sig = sig

        self.callbacks = []
        self.py_times = []
        self.py_inputs = []
        self.py_outputs = []

        mpi_sim_so.create_simulator()

    def load_network(self, filename):
        if isinstance(filename, six.text_type):
            filename = filename.encode('ascii')
        assert isinstance(filename, six.binary_type)
        mpi_sim_so.load_network(filename)

    def finalize_build(self):
        mpi_sim_so.finalize_build()

    def run_n_steps(self, n_steps, progress, log_filename):
        assert isinstance(n_steps, int) and n_steps >= 0
        assert isinstance(progress, bool)

        if isinstance(log_filename, six.text_type):
            log_filename = log_filename.encode('ascii')
        assert isinstance(log_filename, six.binary_type)

        mpi_sim_so.run_n_steps(n_steps, int(progress), log_filename)

    def get_probe_data(self, key):
        assert isinstance(key, int) or isinstance(key, long)
        assert key >= 0
        c_key = ctypes.c_ulonglong(key)

        n_signals = ctypes.c_size_t(0)
        signal_size = ctypes.c_size_t(0)

        c_data = mpi_sim_so.get_probe_data(
            c_key, byref(n_signals), byref(signal_size))
        n_signals = n_signals.value
        signal_size = signal_size.value

        tmp_data = np.ctypeslib.as_array(c_data, (n_signals*signal_size,))

        data = []
        for i in range(n_signals):
            # Important to make a COPY here.
            data.append(tmp_data[i*signal_size:(i+1)*signal_size].copy())

        mpi_sim_so.free_ptr(c_data)

        return data

    def get_signal_value(self, key):
        assert isinstance(key, int) or isinstance(key, long)
        assert key >= 0
        c_key = ctypes.c_ulonglong(key)

        shape1 = ctypes.c_size_t(0)
        shape2 = ctypes.c_size_t(0)

        c_data = mpi_sim_so.get_signal_value(
            c_key, byref(shape1), byref(shape2))
        shape1 = shape1.value
        shape2 = shape2.value

        # Important to make a COPY here
        data = np.ctypeslib.as_array(c_data, (shape1 * shape2,)).copy()

        mpi_sim_so.free_ptr(c_data)

        return data

    def reset(self, seed):
        assert isinstance(seed, int) or isinstance(seed, long)
        assert seed >= 0
        c_seed = ctypes.c_ulonglong(seed)
        mpi_sim_so.reset_simulator(c_seed)

    def close(self):
        mpi_sim_so.close_simulator()

    def create_PyFunc(self, op, index):
        fn = op.fn

        # Handle time.
        pass_time = op.t is not None
        t_signal = op.t if pass_time else self.sig['common'][0]
        t = signal_to_string(t_signal)
        py_time = np.array([22.0])
        self.py_times.append(py_time)
        c_time = py_time.ctypes.data_as(POINTER(c_double))

        # Handle input.
        pass_input = op.x is not None
        input_signal = op.x if pass_input else self.sig['common'][0]
        input = signal_to_string(input_signal)
        if not input_signal.shape:
            py_input = np.array([0.0])
        else:
            py_input = input_signal.initial_value.copy()
        self.py_inputs.append(py_input)
        c_input = py_input.ctypes.data_as(POINTER(c_double))

        # Handle output.
        return_output = op.output is not None
        output_signal = (
            op.output if return_output else self.sig['common']['NULL'])
        output = signal_to_string(output_signal)
        if not output_signal:
            py_output = np.array([0.0])
        else:
            py_output = output_signal.initial_value.copy()
        self.py_outputs.append(py_output)
        c_output = py_output.ctypes.data_as(POINTER(c_double))

        # py_* and c_* should point to the same memory location.

        def py_func():
            # extract time and input from c arrays if applicable
            args = []
            if pass_time:
                args.append(py_time[0])
            if pass_input:
                args.append(py_input)

            y = fn(*args)

            if return_output and y is None:
                # required since Numpy turns None into NaN
                raise SimulationError(
                    "Function %r returned None" % fn.__name__)
            if y is None:
                y = np.array([0.0])

            try:
                float(y[0])
            except:
                try:
                    y = np.array([float(y)])
                except:
                    raise Exception("Cannot use %s as output of Node." % y)

            # store output in c arrays if applicable
            if return_output:
                py_output[:] = y

        # Need to store a handle for the callback so that
        # it doesn't get garbage collected.
        self.callbacks.append(py_func)
        c_func = SIMPLE_PYFUNC(py_func)
        self.callbacks.append(c_func)

        if isinstance(t, six.text_type):
            t = t.encode('ascii')
        if isinstance(input, six.text_type):
            input = input.encode('ascii')
        if isinstance(output, six.text_type):
            output = output.encode('ascii')

        mpi_sim_so.create_PyFunc(
            c_func, t, input, output,
            c_time, c_input, c_output, c_float(index))
