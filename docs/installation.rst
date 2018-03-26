.. _installation:

********************
Installing nengo_mpi
********************

Basic Installation (Ubuntu)
+++++++++++++++++++++++++++
This section outlines how to install nengo_mpi on a typical workstation running Ubuntu.
Other Linux distributions should be able to use these instructions as well, appropriately adapted. If installing
on a high-performance cluster, skip to :ref:`hpc_installation`.

Dependencies
------------
nengo_mpi requires working installations of:

- MPI

- HDF5 (a version compatible with the installed MPI implementation)

- BLAS

- Boost

To satisfy the Boost and BLAS requirements, the following should work on any Ubuntu version after 14.04: ::

    sudo apt-get install libboost-dev libatlas-base-dev

Installing MPI depends on which implementation you want to use:

#. **OpenMPI** - On Ubuntu 14.04 and later the following should be sufficient for getting obtaining OpenMPI and the remaining requirements: ::

    sudo apt-get install openmpi-bin libopenmpi-dev libhdf5-openmpi-dev

#. **MPICH** - On Ubuntu 16.04 (Xenial) the requirements can be satisfied using MPICH with the following invocation: ::

      sudo apt-get install mpich libmpich-dev libhdf5-mpich-dev

   and on Ubuntu 14.04 (Trusty) the following will do the trick: ::

      sudo apt-get install mpich libmpich-dev libhdf5-mpich2-dev

#. **Other MPI Implementations** - May work but have not been tested.

Installation
------------
nengo_mpi is not currently available on PyPI, but can nevertheless be installed via pip once you've downloaded the code. First obtain a copy of the code from github: ::

   git clone https://github.com/nengo/nengo-mpi.git
   cd nengo-mpi

If using the MPICH MPI implementation, we also have to tell the compiler where to find various header files and libraries. Such information is read from the file ``mpi.cfg``. The default ``mpi.cfg`` is setup to work with OpenMPI. We've included configuration files for MPICH in the ``conf`` directory, and making use of them is just a matter of putting them in the right place.

On Ubuntu 16.04 (Xenial): ::

   cp conf/mpich.cfg mpi.cfg

and on Ubuntu 14.04 (Trusty): ::

   cp conf/mpich_trusty.cfg mpi.cfg

Finally install nengo_mpi: ::

   pip install --user .

If you're inside a virtualenv (recommended!) you can omit the ``--user`` flag. If you're developing on nengo_mpi, you can also add the ``-e`` flag so that changes you make to the code will be reflected in your python environment. You can also add ``--install-options="-n"`` to install without building the C++ portion of nengo_mpi, which can be useful when installing nengo_mpi for building network files that you intend to simulate on a different machine.

.. _hpc_installation:

Installing On High-Performance Clusters
+++++++++++++++++++++++++++++++++++++++
High-performance computing environments typically place additional constraints on installing software, so the above installation process may not be available in all such environments. Here we provide some pointers for getting nengo_mpi installed on such environments, with more specific advice for the particular supercomputers that nengo_mpi is most often used on, namely Scinet's General Purpose Cluster and BlueGeneQ.

Dependencies
------------

Installation
------------

With Python
===========

Without Python
==============
