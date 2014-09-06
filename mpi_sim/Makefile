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
OBJS=simulator.o operator.o mpi_operator.o mpi_simulator.o probe.o

DEFS=

SO_LOC=../nengo_mpi
WORKER_LOC=../nengo_mpi

CXXFLAGS= -I${BOOST_INC} -I${MPI_INC} -I$(PYTHON_INC)/python2.7 $(DEFS) -fPIC -std=c++0x

all: mpi_sim

#mpi_sim_worker

run_debug: DEFS+=-DRUN_DEBUG
run_debug: debug

debug: DEFS+=-DDEBUG
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

mpi_operator.o: mpi_operator.cpp mpi_operator.hpp operator.hpp debug.hpp
	mpicxx -std=c++0x -c -o mpi_operator.o -fPIC mpi_operator.cpp -I${BOOST_INC} -I${MPI_INC}

mpi_simulator.o: mpi_simulator.cpp mpi_simulator.hpp simulator.hpp
	mpicxx -std=c++0x -c -o mpi_simulator.o -fPIC mpi_simulator.cpp -I${BOOST_INC} -I${MPI_INC}

probe.o: probe.cpp probe.hpp operator.hpp
python.o: python.cpp python.hpp simulator.hpp operator.hpp mpi_operator.hpp probe.hpp debug.hpp
operator.o: operator.cpp operator.hpp debug.hpp
simulator.o: simulator.cpp simulator.hpp mpi_operator.hpp operator.hpp probe.hpp debug.hpp


clean:
	rm -rf ${SO_LOC}/mpi_sim.so ${WORKER_LOC}/mpi_sim_worker *.o


