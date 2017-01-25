#!/bin/sh
set -e
sudo apt-get -qq update
case $1 in
  mpich) set -x;
    sudo apt-get install -y -q mpich libmpich-dev libhdf5-mpich-dev libboost-dev
    cp conf/mpich.cfg mpi.cfg
    ;;
  openmpi) set -x;
    sudo apt-get install -y -q openmpi-bin libopenmpi-dev libhdf5-openmpi-dev libboost-dev
    cp conf/openmpi.cfg mpi.cfg
    ;;
  *)
    echo "Unknown MPI implementation:" $1
    exit 1
    ;;
esac
