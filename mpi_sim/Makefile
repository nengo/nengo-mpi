HOST=$(shell uname --nodename)

ifneq (, $(findstring bgq,$(HOST)))
	#on bgq
	CXX=mpicxx
	PYTHON_INC=${SCINET_PYTHON_INC}/python2.7
	BOOST_INC=${SCINET_BOOST_INC}
	BOOST_LIB=${SCINET_BOOST_LIB}
	PYTHON_LIB=${SCINET_PYTHON_LIB}
	STD=c++0x # Redhat 4.4.7, which we have access to on bgq, calls it c++0x
else ifeq (, $(findstring gpc,$(HOST)))
	#on gpc
	CXX=mpicxx
	PYTHON_INC=${SCINET_PYTHON_INC}
	BOOST_INC=${SCINET_BOOST_INC}
	BOOST_LIB=${SCINET_BOOST_LIB}
	PYTHON_LIB=${SCINET_PYTHON_LIB}
	STD=c++11
else
	#on other machine
	CXX=g++
	PYTHON_INC=/usr/include
	BOOST_INC=
	BOOST_LIB=${SCINET_BOOST_LIB}
endif

OBJS=simulator.o operator.o mpi_operator.o custom_ops.o probe.o chunk.o mpi_interface.o

SO_LOC=${HOME}/nengo_mpi/nengo_mpi

WORKER_LOC=${HOME}/nengo_mpi/nengo_mpi

DEFS=-O3

all: DEFS += -DBOOST_UBLAS_NDEBUG -DNDEBUG
all: build

run_debug: DEFS+=-DRUN_DEBUG
run_debug: debug

debug: DEFS+=-DDEBUG -g
debug: build

CXXFLAGS= -I${BOOST_INC} -I${PYTHON_INC} ${DEFS} -fPIC -std=${STD}

build: mpi_sim_worker mpi_sim

clean:
	rm -rf ${SO_LOC}/mpi_sim.so ${WORKER_LOC}/mpi_sim_worker *.o

# ********* mpi_sim_worker *************

mpi_sim_worker: worker.o ${OBJS}
	${CXX} -o ${WORKER_LOC}/mpi_sim_worker worker.o ${OBJS} ${DEFS} -std=${STD} -L${BOOST_LIB} -lm

worker.o: worker.cpp simulator.hpp operator.hpp mpi_operator.hpp probe.hpp debug.hpp
#	${CXX} -c -o worker.o worker.cpp -std=${STD} -I${BOOST_INC} ${DEFS}

# ********* mpi_sim.so *************

mpi_sim: ${OBJS} python.o
	${CXX} -o ${SO_LOC}/mpi_sim.so ${OBJS} python.o -shared ${DEFS} -std=${STD} -L${BOOST_LIB} -lm -L${PYTHON_LIB} -lboost_python

# mpi_operator.o: mpi_operator.cpp mpi_operator.hpp operator.hpp debug.hpp
# 	mpicxx -std=${STD} -c -o mpi_operator.o -fPIC mpi_operator.cpp -I${BOOST_INC} ${DEFS}
#
# mpi_interface.o: mpi_interface.cpp mpi_interface.hpp simulator.hpp
# 	mpicxx -std=${STD} -c -o mpi_interface.o -fPIC mpi_interface.cpp -I${BOOST_INC} ${DEFS}

mpi_operator.o: mpi_operator.cpp mpi_operator.hpp operator.hpp debug.hpp
mpi_interface.o: mpi_interface.cpp mpi_interface.hpp simulator.hpp
probe.o: probe.cpp probe.hpp operator.hpp debug.hpp
python.o: python.cpp python.hpp simulator.hpp chunk.hpp operator.hpp mpi_operator.hpp probe.hpp debug.hpp
operator.o: operator.cpp operator.hpp debug.hpp
chunk.o: chunk.cpp chunk.hpp operator.hpp mpi_operator.hpp custom_ops.hpp probe.hpp debug.hpp
simulator.o: simulator.cpp simulator.hpp chunk.hpp mpi_interface.hpp debug.hpp
custom_ops.o: custom_ops.cpp custom_ops.hpp operator.hpp debug.hpp
