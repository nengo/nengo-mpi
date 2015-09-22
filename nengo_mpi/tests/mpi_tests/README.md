All scripts in this directory (whose names end with `.py`) will be executed as tests
using the function nengo_mpi.tests.test\_mpi.test\_mpi\_script. Each script will be
run in the context:

mpirun -np \<n\_procs\> python -m nengo\_mpi \<script\_name\>

where n\_procs is the number of processors, given as an argument to test\_mpi\_script.