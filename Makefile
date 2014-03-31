
CPP=icpc

nengo_mpi.so: operator.o simulator.o
	${CPP} -o nengo_mpi.so operator.o simulator.o -shared -lboost_python -lm

operator: operator.hpp operator.cpp
	${CPP} -c operator.o operator.cpp -fPIC

simulator: simulator.hpp simulator.cpp
	${CPP} -c simulator simulator.cpp -fPIC

clean:
	rm -rf nengo_mpi.so *.o
