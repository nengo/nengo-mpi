import numpy as np
from collections import OrderedDict


OP_DELIM = ";"
SIGNAL_DELIM = ":"
PROBE_DELIM = "|"
STAND_IN = "-"
assert (
    STAND_IN != SIGNAL_DELIM
    and STAND_IN != OP_DELIM
    and STAND_IN != PROBE_DELIM)


def make_key(obj):
    """ Create a unique key for an object.

    Must be reproducable (i.e. produce the same key if called with
    the same object multiple times).

    """
    return id(obj)


def sanitize_label(s):
    s = s.replace(SIGNAL_DELIM, STAND_IN)
    s = s.replace(OP_DELIM, STAND_IN)
    s = s.replace(PROBE_DELIM, STAND_IN)

    return s


def pad(x):
    return (
        (1, 1) if len(x) == 0 else (
            (x[0], 1) if len(x) == 1 else x))


def signal_to_string(signal, debug=False):
    """ Convert a signal to a string.

    The format of the returned string is:
        signal_key:label:ndim:shape0,shape1:stride0,stride1:offset

    """
    shape = pad(signal.shape)
    stride = pad(signal.elemstrides)

    signal_args = [
        make_key(signal.base),
        sanitize_label(signal.name) if debug else '',
        signal.ndim,
        "%d,%d" % (shape[0], shape[1]),
        "%d,%d" % (stride[0], stride[1]),
        signal.elemoffset
    ]

    signal_string = SIGNAL_DELIM.join(map(str, signal_args))
    return signal_string


def ndarray_to_string(a):
    s = "%d,%d," % np.atleast_2d(a).shape
    s += ",".join([str(n) for n in a.flatten()])
    return s


# Stole this from nengo_ocl
def get_closures(f):
    return OrderedDict(zip(
        f.__code__.co_freevars, (c.cell_contents for c in f.__closure__)))
