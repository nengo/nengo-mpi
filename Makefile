
CPP=icpc

nengo_mpi.so: operator.o simulator.o
	${CPP} -o nengo_mpi.so operator.o simulator.o -shared -lboost_python -lm

operator.o: operator.hpp operator.cpp
	${CPP} -c -o operator.o operator.cpp -fPIC

simulator.o: simulator.hpp simulator.cpp
	${CPP} -c -o simulator.o simulator.cpp -fPIC

clean:
	rm -rf nengo_mpi.so *.o
