Most tests in nengo\_mpi are executed with a single processor. This is done
because it is not clear how to make pytest play nicely with having multiple
processors running. It may be possible, but a consistent way to do it has not
yet been implemented. Thus, in order to get at least some testing nengo\_mpi
with multiple processors (which is, of course, its main use-case), a number
of tests have been created which launch scripts using nengo\_mpi with multiple
processors.

All scripts in this directory (whose names end with `.py`) will be executed as tests
using the function nengo_mpi.tests.test\_mpi.test\_mpi\_script. Each script will be
run in the context:

mpirun -np \<n\_procs\> python -m nengo\_mpi \<script\_name\>

where n\_procs is the number of processors, given as an argument to test\_mpi\_script.
Tests are considered to pass if they return a 0 exit code.