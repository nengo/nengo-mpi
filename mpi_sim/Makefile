HOST=$(shell uname --nodename)

ifneq (, $(findstring bgq,$(HOST)))
	#on bgq
	CXX=mpicxx
	BOOST_INC=${SCINET_BOOST_INC}
	BOOST_LIB=${SCINET_BOOST_LIB}
	PYTHON_INC=${SCINET_PYTHON_INC}/python2.7
	PYTHON_LIB=${SCINET_PYTHON_LIB}
	STD=c++0x # Redhat 4.4.7, which we have access to on bgq, calls it c++0x
else ifneq (, $(findstring gpc,$(HOST)))
	#on gpc
	CXX=mpicxx
	BOOST_INC=${SCINET_BOOST_INC}
	BOOST_LIB=${SCINET_BOOST_LIB}
	PYTHON_INC=${SCINET_PYTHON_INC}/python2.7
	PYTHON_LIB=${SCINET_PYTHON_LIB}
	STD=c++11
else
	#on other machine
	CXX=g++
	PYTHON_INC=/usr/include
	BOOST_INC=
	BOOST_LIB=${SCINET_BOOST_LIB}
endif

OBJS=simulator.o operator.o mpi_operator.o custom_ops.o probe.o chunk.o mpi_simulator.o sim_log.o utils.o

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
	${CXX} -o ${WORKER_LOC}/mpi_sim_worker worker.o ${OBJS} ${DEFS} -std=${STD} -L${BOOST_LIB} -lm -lhdf5

worker.o: worker.cpp simulator.hpp operator.hpp mpi_operator.hpp probe.hpp utils.hpp debug.hpp

# ********* mpi_sim.so *************

mpi_sim: ${OBJS} python.o
	${CXX} -o ${SO_LOC}/mpi_sim.so ${OBJS} python.o -shared ${DEFS} -std=${STD} -L${BOOST_LIB} -lm -L${PYTHON_LIB} -lboost_python -lhdf5

mpi_operator.o: mpi_operator.cpp mpi_operator.hpp operator.hpp debug.hpp
mpi_simulator.o: mpi_simulator.cpp mpi_simulator.hpp simulator.hpp
probe.o: probe.cpp probe.hpp operator.hpp debug.hpp
python.o: python.cpp python.hpp simulator.hpp chunk.hpp operator.hpp mpi_operator.hpp probe.hpp debug.hpp
operator.o: operator.cpp operator.hpp debug.hpp
chunk.o: chunk.cpp chunk.hpp operator.hpp mpi_operator.hpp custom_ops.hpp probe.hpp debug.hpp sim_log.hpp
simulator.o: simulator.cpp simulator.hpp chunk.hpp mpi_simulator.hpp debug.hpp
custom_ops.o: custom_ops.cpp custom_ops.hpp operator.hpp debug.hpp
sim_log.o: sim_log.cpp operator.hpp debug.hpp
utils.o: utils.cpp utils.hpp debug.hpp
