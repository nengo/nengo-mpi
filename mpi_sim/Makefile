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

OBJS=mpi.o probe.o operator.o simulator.o python.o 

DEFS=
#DEFS=-DBOOST_UBLAS_TYPE_CHECK=0

SO_LOC=../nengo_mpi

all: mpi_sim

debug: DEFS+=-D_DEBUG
debug: mpi_sim

CXXFLAGS= -I${MPI_INC} -I$(PYTHON_INC)/python2.7 $(DEFS) -fPIC -std=c++0x
#-I$(BOOST_INC) 

mpi_sim: ${OBJS}
	${CXX} -o ${SO_LOC}/mpi_sim.so ${OBJS} -shared -L${BOOST_LIB} -lboost_python -lboost_serialization -lm

operator.o: operator.hpp operator.cpp
simulator.o: simulator.hpp simulator.cpp
probe.o: probe.hpp probe.cpp
python.o: python.cpp python.hpp
mpi.o: mpi.cpp mpi.hpp
	mpicxx -std=c++0x -c -o mpi.o -fPIC mpi.cpp 

clean:
	rm -rf ${SO_LOC}/mpi_sim.so *.o
