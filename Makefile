
CPP=icpc

DEFS=

all: nengo_mpi.so

debug: DEFS+=-D_DEBUG
debug: nengo_mpi.so

nengo_mpi.so: operator.o simulator.o python.o 
	${CPP} -o nengo_mpi.so operator.o simulator.o python.o -shared -lboost_python -lm 

operator.o: operator.hpp operator.cpp
	${CPP} -c -o operator.o operator.cpp -fPIC ${DEFS}

simulator.o: simulator.hpp simulator.cpp
	${CPP} -c -o simulator.o simulator.cpp -fPIC ${DEFS}

python.o: python.cpp python.hpp
	${CPP} -c -o python.o python.cpp -fPIC -I${SCINET_PYTHON_INC}/python2.7 ${DEFS}

clean:
	rm -rf nengo_mpi.so *.o
