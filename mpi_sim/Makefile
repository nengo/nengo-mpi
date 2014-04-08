
CPP=icpc

DEFS=

SO_LOC=../nengo_mpi

all: mpi_sim

debug: DEFS+=-D_DEBUG
debug: mpi_sim

mpi_sim: operator.o simulator.o python.o probe.o
	${CPP} -o ${SO_LOC}/mpi_sim.so probe.o operator.o simulator.o python.o -shared -lboost_python -lm 

operator.o: operator.hpp operator.cpp
	${CPP} -c -o operator.o operator.cpp -fPIC ${DEFS}

simulator.o: simulator.hpp simulator.cpp
	${CPP} -c -o simulator.o simulator.cpp -fPIC ${DEFS}

probe.o: probe.hpp probe.cpp
	${CPP} -c -o probe.o probe.cpp -fPIC ${DEFS}

python.o: python.cpp python.hpp
	${CPP} -c -o python.o python.cpp -fPIC -I${SCINET_PYTHON_INC}/python2.7 ${DEFS}

clean:
	rm -rf ${SO_LOC}/mpi_sim.so *.o
