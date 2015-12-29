"""
nengo_mpi
=========

nengo_mpi provides an MPI backend for the Nengo neural simulation package.

The source code repository for this package is found at
https://www.github.com/e2crawfo/nengo_mpi.
"""

__copyright__ = "2013-2014, Applied Brain Research"
__license__ = "Free for non-commercial use; see LICENSE.rst"
from .version import version as __version__

import logging


class NengoMpiException(Exception):
    pass


class PartitionError(NengoMpiException):
    pass


from nengo_mpi.simulator import Simulator
from nengo_mpi.partition import Partitioner
from nengo_mpi.spaun_mpi import SpaunStimulus

logger = logging.getLogger(__name__)
try:
    # Prevent output if no handler set
    logger.addHandler(logging.NullHandler())
except AttributeError:
    pass