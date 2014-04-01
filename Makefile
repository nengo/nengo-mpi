
CPP=icpc

nengo_mpi.so: operator.o simulator.o python.o
	${CPP} -o nengo_mpi.so operator.o simulator.o python.o -shared -lboost_python -lm

operator.o: operator.hpp operator.cpp
	${CPP} -c -o operator.o operator.cpp -fPIC

simulator.o: simulator.hpp simulator.cpp
	${CPP} -c -o simulator.o simulator.cpp -fPIC

python.o: python.cpp python.hpp
	${CPP} -c -o python.o python.cpp -fPIC -I${SCINET_PYTHON_INC}/python2.7

clean:
	rm -rf nengo_mpi.so *.o
