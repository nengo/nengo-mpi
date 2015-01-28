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

OBJS=simulator.o operator.o mpi_operator.o probe.o chunk.o mpi_interface.o

SO_LOC=../nengo_mpi

WORKER_LOC=/home/c/celiasmi/e2crawfo/nengo_mpi/nengo_mpi

DEFS=
#DEFS=-O3

STD=c++11
#STD=c++0x

all: build

run_debug: DEFS+=-DRUN_DEBUG
run_debug: debug

debug: DEFS+=-DDEBUG -g
debug: build

CXXFLAGS= -I${BOOST_INC} -I${MPI_INC} -I$(PYTHON_INC)/python2.7 $(DEFS) -fPIC -std=${STD}

build: mpi_sim_worker mpi_sim

clean:
	rm -rf ${SO_LOC}/mpi_sim.so ${WORKER_LOC}/mpi_sim_worker *.o

# ********* mpi_sim_worker *************
# Worker doesn't need any of the python includes.

mpi_sim_worker: worker.o ${OBJS}
	${CXX} -o ${WORKER_LOC}/mpi_sim_worker worker.o ${OBJS} -L${MPI_LIB} -L${BOOST_LIB} -lmpi -lm -std=${STD} ${DEFS}

worker.o: worker.cpp simulator.hpp operator.hpp mpi_operator.hpp probe.hpp debug.hpp
	mpicxx -std=${STD} -c -o worker.o worker.cpp -I${BOOST_INC} -I${MPI_INC} ${DEFS}

# ********* mpi_sim.so *************

mpi_sim: ${OBJS} python.o
	${CXX} -o ${SO_LOC}/mpi_sim.so ${OBJS} python.o -shared -L${PYTHON_LIB} -L${MPI_LIB} -L${BOOST_LIB} -lboost_python -lmpi -lm ${DEFS}

mpi_operator.o: mpi_operator.cpp mpi_operator.hpp operator.hpp debug.hpp
	mpicxx -std=${STD} -c -o mpi_operator.o -fPIC mpi_operator.cpp -I${BOOST_INC} -I${MPI_INC} ${DEFS}

mpi_interface.o: mpi_interface.cpp mpi_interface.hpp simulator.hpp
	mpicxx -std=${STD} -c -o mpi_interface.o -fPIC mpi_interface.cpp -I${BOOST_INC} -I${MPI_INC} ${DEFS}

probe.o: probe.cpp probe.hpp operator.hpp
python.o: python.cpp python.hpp simulator.hpp chunk.hpp operator.hpp mpi_operator.hpp probe.hpp debug.hpp
operator.o: operator.cpp operator.hpp debug.hpp
chunk.o: chunk.cpp chunk.hpp operator.hpp mpi_operator.hpp probe.hpp debug.hpp
simulator.o: simulator.cpp simulator.hpp chunk.hpp mpi_interface.hpp debug.hpp
