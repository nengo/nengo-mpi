HOST=$(shell uname --nodename)

ifneq (, $(findstring gpc,$(HOST)))
	#on gpc
	CXX=icpc
	PYTHON_INC=${SCINET_PYTHON_INC}
	BOOST_INC=${SCINET_BOOST_INC}
	MPI_INC=${SCINET_MPI_INC}
	BOOST_LIB=${SCINET_BOOST_LIB}
	PYTHON_LIB=${SCINET_PYTHON_LIB}
	MPI_LIB=${SCINET_MPI_LIB}
else
	#on other machine
	CXX=g++
	PYTHON_INC=/usr/include
	BOOST_INC=
	MPI_INC=${SCINET_MPI_INC}
	BOOST_LIB=${SCINET_BOOST_LIB}
endif

# Doesn't contain python.o
OBJS=simulator.o operator.o mpi_operators.o mpi_simulator.o probe.o

DEFS=

SO_LOC=../nengo_mpi
WORKER_LOC=../nengo_mpi

CXXFLAGS= -I${BOOST_INC} -I${MPI_INC} -I$(PYTHON_INC)/python2.7 $(DEFS) -fPIC -std=c++0x

all: mpi_sim mpi_sim_worker

run_debug: DEFS+=-D_RUN_DEBUG
run_debug: debug

debug: DEFS+=-D_DEBUG
debug: CXXFLAGS+= -g
debug: mpi_sim

# ********* Worker *************
mpi_sim_worker: worker.o ${OBJS}
	${CXX} -o ${WORKER_LOC}/mpi_sim_worker worker.o ${OBJS} -L${BOOST_LIB} -lboost_mpi -lboost_serialization -lm -std=c++0x

worker.o: worker.cpp
	mpicxx -c -o worker.o worker.cpp -I${SCINET_BOOST_INC} ${CXXFLAGS}


# ********* Shared library *************
mpi_sim: ${OBJS} python.o
	${CXX} -o ${SO_LOC}/mpi_sim.so ${OBJS} python.o -shared -L${PYTHON_LIB} -L${MPI_LIB} -L${BOOST_LIB} -lboost_serialization -lboost_python -lmpi -lboost_mpi -lm

mpi_operators.o: mpi_operators.cpp mpi_operators.hpp
	mpicxx -std=c++0x -c -o mpi_operators.o -fPIC mpi_operators.cpp -I${BOOST_INC} -I${MPI_INC}

mpi_simulator.o: mpi_simulator.cpp mpi_simulator.hpp
	mpicxx -std=c++0x -c -o mpi_simulator.o -fPIC mpi_simulator.cpp -I${BOOST_INC} -I${MPI_INC}

probe.o: probe.hpp probe.cpp
python.o: python.cpp python.hpp
operator.o: operator.hpp operator.cpp
simulator.o: simulator.hpp simulator.cpp


clean:
	rm -rf ${SO_LOC}/mpi_sim.so ${WORKER_LOC}/mpi_sim_worker *.o


