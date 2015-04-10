# A sample submission script for submitting a job on gpc.


#!/bin/bash
# MOAB/Torque submission script for SciNet GPC 
#
#PBS -l nodes=3:ppn=8,walltime=0:15:00
#PBS -N nengo_mpi
#PBS -m abe

# load modules (must match modules used for compilation)
module load intel/14.0.1
module load python/2.7.5
module load openmpi/intel/1.6.4
module load cxxlibraries/boost/1.55.0-intel
module load gcc/4.8.1
module load use.own
module load nengo

# DIRECTORY TO RUN - $PBS_O_WORKDIR is directory job was submitted from
cd /home/c/celiasmi/e2crawfo/nengo_mpi/scripts

# EXECUTION COMMAND;
mpirun -np 1 python benchmark.py --noprog > /scratch/c/celiasmi/e2crawfo/experiments/temp.txt