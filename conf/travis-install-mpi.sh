#!/bin/sh
set -e
sudo apt-get -qq update
case $1 in
  mpich) set -x;
    sudo apt-get install -y -q mpich libmpich-dev libhdf5-mpich2-dev libboost-dev libatlas-base-dev  # On trusty
    # sudo apt-get install -y -q mpich libmpich-dev libhdf5-mpich-dev libboost-dev libblas-base-dev # On xenial
    cp conf/mpich_trusty.cfg mpi.cfg
    ;;
  openmpi) set -x;
    sudo apt-get install -y -q openmpi-bin libopenmpi-dev libhdf5-openmpi-dev libboost-dev libatlas-base-dev
    cp conf/openmpi_trusty.cfg mpi.cfg
    ;;
  *)
    echo "Unknown MPI implementation:" $1
    exit 1
    ;;
esac
