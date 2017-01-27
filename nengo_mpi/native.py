import six
import numpy as np
import ctypes

from nengo.exceptions import SimulationError

from nengo_mpi.utils import signal_to_string

_native_sim_available = False
try:
    import mpi_sim
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
        self.time_buffers = []
        self.input_buffers = []
        self.output_buffers = []

        mpi_sim.create_simulator()

    def load_network(self, filename):
        assert isinstance(filename,
                          six.text_type if six.PY3 else six.binary_type)
        mpi_sim.load_network(filename)

    def finalize_build(self):
        mpi_sim.finalize_build()

    def run_n_steps(self, n_steps, progress, log_filename):
        assert isinstance(n_steps, int) and n_steps >= 0
        assert isinstance(progress, bool)
        assert isinstance(log_filename,
                          six.text_type if six.PY3 else six.binary_type)

        mpi_sim.run_n_steps(n_steps, int(progress), log_filename)

    def get_probe_data(self, key):
        assert isinstance(key, int) or isinstance(key, long)
        assert key >= 0
        return mpi_sim.get_probe_data(key)

    def get_signal_value(self, key):
        assert isinstance(key, int) or isinstance(key, long)
        assert key >= 0
        return mpi_sim.get_signal_value(key)

    def reset(self, seed):
        assert isinstance(seed, int) or isinstance(seed, long)
        assert seed >= 0
        mpi_sim.reset_simulator(seed)

    def close(self):
        mpi_sim.close_simulator()

    def create_PyFunc(self, op, index):
        fn = op.fn

        # Handle time.
        pass_time = op.t is not None
        t_signal = op.t if pass_time else self.sig['common'][0]
        t_string = signal_to_string(t_signal)
        time_buffer = np.array([22.0])
        self.time_buffers.append(time_buffer)

        # Handle input.
        pass_input = op.x is not None
        input_signal = op.x if pass_input else self.sig['common'][0]
        input_string = signal_to_string(input_signal)
        if not input_signal.shape:
            input_buffer = np.array([0.0])
        else:
            input_buffer = input_signal.initial_value.copy()
        self.input_buffers.append(input_buffer)

        # Handle output.
        return_output = op.output is not None
        output_signal = (
            op.output if return_output else self.sig['common']['NULL'])
        output_string = signal_to_string(output_signal)
        if not output_signal:
            output_buffer = np.array([0.0])
        else:
            output_buffer = output_signal.initial_value.copy()
        self.output_buffers.append(output_buffer)

        def py_func():
            # extract time and input if applicable
            args = []
            if pass_time:
                args.append(time_buffer[0])
            if pass_input:
                args.append(input_buffer)

            y = fn(*args)

            if return_output and y is None:
                # required since Numpy turns None into NaN
                raise SimulationError(
                    "Function %r returned None." % fn.__name__)
            if y is None:
                y = np.array([0.0])

            try:
                float(y[0])
            except:
                try:
                    y = np.array([float(y)])
                except:
                    raise Exception("Cannot use %s as output of Node." % y)

            # store output if applicable
            if return_output:
                output_buffer[:] = y

        # Need to store a handle for the callback so that
        # it doesn't get garbage collected.
        self.callbacks.append(py_func)
        for s in [t_string, input_string, output_string]:
            assert isinstance(s, six.text_type if six.PY3 else six.binary_type)

        mpi_sim.create_PyFunc(
            py_func, t_string, input_string, output_string,
            time_buffer, input_buffer, output_buffer, index)
