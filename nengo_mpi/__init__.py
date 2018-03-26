"""
nengo_mpi
=========

nengo_mpi provides an MPI backend for the Nengo neural simulation package.

The source code repository for this package is found at
https://www.github.com/nengo/nengo-mpi.
"""

from .__about__ import (
    __author__, __commit__, __copyright__, __email__, __license__, __summary__,
    __title__, __uri__, __version__,
)
version = __version__


class NengoMpiException(Exception):
    pass


class PartitionError(NengoMpiException):
    pass


from .simulator import Simulator
from .partition import Partitioner
from .spaun_mpi import SpaunStimulus

import logging
logger = logging.getLogger(__name__)
try:
    # Prevent output if no handler set
    logger.addHandler(logging.NullHandler())
except AttributeError:
    pass
