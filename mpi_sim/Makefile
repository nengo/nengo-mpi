HOST=$(shell uname --nodename)

ifneq (, $(findstring gpc,$(HOST)))
	#on gpc
	#CPP=icpc
	CXX=icpc
	PYTHON_INC=${SCINET_PYTHON_INC}
	BOOST_INC=${SCINET_BOOST_INC}
	MPI_INC=${SCINET_MPI_INC}
	BOOST_LIB=${SCINET_BOOST_LIB}
else
	#on other machine
	#CPP=g++
	CXX=g++
	PYTHON_INC=/usr/include
	BOOST_INC=
	MPI_INC=${SCINET_MPI_INC}
	BOOST_LIB=${SCINET_BOOST_LIB}
endif

OBJS=simulator.o operator.o mpi_operators.o mpi_simulator.o probe.o python.o 

DEFS=
#DEFS=-DBOOST_UBLAS_TYPE_CHECK=0

SO_LOC=../nengo_mpi
WORKER_LOC=../nengo_mpi

all: mpi_sim mpi_sim_worker

debug: DEFS+=-D_DEBUG
debug: mpi_sim

CXXFLAGS= -I${MPI_INC} -I$(PYTHON_INC)/python2.7 $(DEFS) -fPIC -std=c++0x

#Worker code
mpi_sim_worker: worker.cpp
	mpicxx -o ${WORKER_LOC}/mpi_sim_worker worker.cpp -I${SCINET_BOOST_INC} -L${SCINET_BOOST_LIB} -lboost_mpi -lboost_serialization -std=c++0x


#Main library code
mpi_sim: ${OBJS}
	${CXX} -o ${SO_LOC}/mpi_sim.so ${OBJS} -shared -L${BOOST_LIB} -lboost_python -lboost_serialization -lm

mpi_operators.o: mpi_operators.cpp mpi_operators.hpp
	mpicxx -std=c++0x -c -o mpi_operators.o -fPIC mpi_operators.cpp 

mpi_simulator.o: mpi_simulator.cpp mpi_simulator.hpp
	mpicxx -std=c++0x -c -o mpi_simulator.o -fPIC mpi_simulator.cpp 

probe.o: probe.hpp probe.cpp
python.o: python.cpp python.hpp
operator.o: operator.hpp operator.cpp
simulator.o: simulator.hpp simulator.cpp

clean:
	rm -rf ${SO_LOC}/mpi_sim.so ${WORKER_LOC}/mpi_sim_worker *.o 
