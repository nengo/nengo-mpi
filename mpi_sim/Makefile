HOST=$(shell uname --nodename)

ifneq (, $(findstring gpc,$(HOST)))
	#on gpc
	CPP=icpc
	PYTHON_INC=${SCINET_PYTHON_INC}
else
	#on other machine
	CPP=g++
	PYTHON_INC=/usr/include
endif


DEFS=
#DEFS=-DBOOST_UBLAS_TYPE_CHECK=0

SO_LOC=../nengo_mpi

all: mpi_sim

debug: DEFS+=-D_DEBUG
debug: mpi_sim

CXXFLAGS= -I$(PYTHON_INC)/python2.7 $(DEFS) -fPIC

mpi_sim: operator.o simulator.o python.o probe.o
	${CPP} -o ${SO_LOC}/mpi_sim.so probe.o operator.o simulator.o python.o -shared -lboost_python -lm

operator.o: operator.hpp operator.cpp
simulator.o: simulator.hpp simulator.cpp
probe.o: probe.hpp probe.cpp
python.o: python.cpp python.hpp

clean:
	rm -rf ${SO_LOC}/mpi_sim.so *.o
