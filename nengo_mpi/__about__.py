import os

__all__ = [
    "__title__", "__summary__", "__uri__", "__version__", "__commit__",
    "__author__", "__email__", "__license__", "__copyright__",
]


try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = None


__title__ = "nengo_mpi"
__summary__ = "MPI backend for the nengo neural simulator."
__uri__ = "https://nengo-mpi.readthedocs.io"

if base_dir is not None and os.path.exists(os.path.join(base_dir, ".commit")):
    with open(os.path.join(base_dir, ".commit")) as fp:
        __commit__ = fp.read().strip()
else:
    __commit__ = None

__author__ = "Eric Crawford"
__email__ = "eric.crawford@mail.mcgill.ca"

__copyright__ = "2014-2017, Applied Brain Research"
__license__ = "Free for non-commercial use; see LICENSE.rst"

"""nengo_mpi version information.

We use semantic versioning (see http://semver.org/).
and confrom to PEP440 (see https://www.python.org/dev/peps/pep-0440/).
'.devN' will be added to the version unless the code base represents
a release version. Release versions are git tagged with the version.
"""
version_info = (0, 1, 0)  # (major, minor, patch)
dev = 0
__version__ = "{v}{dev}".format(v='.'.join(str(v) for v in version_info),
                            dev=('.dev%d' % dev) if dev is not None else '')

