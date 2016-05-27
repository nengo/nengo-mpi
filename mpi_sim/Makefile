HOST=$(shell uname --nodename)

ifneq (, $(findstring bgq,$(HOST)))
	#on bgq
	MPICXX=mpicxx
	CXX=${MPICXX}
	BOOST_INC=-I${SCINET_BOOST_INC}
	BOOST_LIB=-L${SCINET_BOOST_LIB}
	CBLAS_INC=
	CBLAS_LIB=
	PYTHON_INC=
	PYTHON_LIB=
	HDF5_INC=-I${SCINET_HDF5_INC}
	HDF5_LIB=-L${SCINET_HDF5_LIB}
	COMPRESSION_LIBS= -L${SCINET_ZLIB_LIB} -L${SCINET_SZIP_LIB} -lsz -lz
	STD=c++0x # Redhat 4.4.7, which we use on bgq, uses the name c++0x for c++11
	CXXFLAGS= ${BOOST_INC} ${HDF5_INC} ${DEFS} -std=${STD}
	DO_PYTHON=FALSE
else ifneq (, $(findstring gpc,$(HOST)))
	#on gpc
	MPICXX=mpicxx
	CXX=${MPICXX}
	BOOST_INC=-I${SCINET_BOOST_INC}
	BOOST_LIB=-L${SCINET_BOOST_LIB}
	CBLAS_INC=
	CBLAS_LIB=
	PYTHON_INC=-I${SCINET_PYTHON_INC}/python2.7
	PYTHON_LIB=-L${SCINET_PYTHON_LIB}
	HDF5_INC=-I${SCINET_HDF5_INC}
	HDF5_LIB=-L${SCINET_HDF5_LIB}
	COMPRESSION_LIBS=
	STD=c++11
	CXXFLAGS= ${PYTHON_INC} ${BOOST_INC} ${HDF5_INC} ${DEFS} -fPIC -std=${STD}
	DO_PYTHON=TRUE
else ifneq (, $(findstring comet,$(HOST)))
	#on gpc
	MPICXX=mpicxx
	CXX=${MPICXX}
	BOOST_INC=-I${BOOST_ROOT}/include
	BOOST_LIB=-L${BOOST_ROOT}/lib
	CBLAS_INC=-I${ATLAS_HOME}/include
	CBLAS_LIB=# found automatically -L${ATLAS_HOME}/lib
	PYTHON_INC=-I${PYTHON_ROOT}/include/python2.7
	PYTHON_LIB=# found automatically -L${PYTHON_ROOT}/include
	HDF5_INC=-I${HDF5HOME}/include
	HDF5_LIB=-L${HDF5HOME}/lib
	COMPRESSION_LIBS=
	STD=c++11
	CXXFLAGS= ${PYTHON_INC} ${BOOST_INC} ${HDF5_INC} ${DEFS} -fPIC -std=${STD}
	DO_PYTHON=TRUE
else
	#on other machine
	MPICXX=mpicxx
	CXX=${MPICXX}
	BOOST_INC=-I/usr/include/boost/
	BOOST_LIB=
	CBLAS_INC=-I/usr/include/atlas/
	CBLAS_LIB=-L/usr/lib/atlas-base/
	PYTHON_INC=-I/usr/include/python2.7/
	PYTHON_LIB=
	HDF5_LIB=-L/usr/lib/x86_64-linux-gnu/
	HDF5_INC=-I/usr/include/hdf5/openmpi/
	STD=c++11
	CXXFLAGS= ${CBLAS_INC} ${PYTHON_INC} ${BOOST_INC} ${HDF5_INC} ${DEFS} -fPIC -std=${STD}
	DO_PYTHON=TRUE
endif

OBJS=signal.o operator.o simulator.o spec.o spaun.o probe.o chunk.o sim_log.o debug.o utils.o
MPI_OBJS=${OBJS} mpi_simulator.o mpi_operator.o psim_log.o
BIN=${HOME}/nengo_mpi/bin

# If building on a cluster we don't need to build mpi_sim.so, which is only used for python.
ifeq (${DO_PYTHON}, TRUE)
	MPI_SIM_SO=mpi_sim.so
else
	MPI_SIM_SO=
endif

DEFS=
#DEFS=-Wconversion -Wall

all: DEFS += -DNDEBUG -O3
all: build

# Print simulation-related debug info.
run_dbg: DEFS+= -DRUN_DEBUG
run_dbg: mpi_dbg

# Print build-related debug info.
build_dbg: DEFS+= -DBUILD_DEBUG
build_dbg: dbg

# Print mpi-related debug info.
mpi_dbg: DEFS+= -DMPI_DEBUG
mpi_dbg: dbg

# All special debugging modes also activate basic debugging output.
# Each MPI process will direct its output to a file called
# chunk_x_dbg, where x is the rank of the processor.
dbg: DEFS+= -DDEBUG -g
dbg: build

build: nengo_cpp nengo_mpi ${MPI_SIM_SO}

clean:
	rm -rf ${BIN}/nengo_cpp ${BIN}/nengo_mpi ${BIN}/mpi_sim.so *.o


# ********* nengo_cpp *************

nengo_cpp: nengo_cpp.o ${MPI_OBJS} | ${BIN}
	${CXX} -o ${BIN}/nengo_cpp nengo_cpp.o ${MPI_OBJS} ${DEFS} -std=${STD} ${CBLAS_LIB} -lcblas ${BOOST_LIB} -lm ${HDF5_LIB} -lhdf5_openmpi -ldl  ${COMPRESSION_LIBS}

nengo_cpp.o: nengo_mpi.cpp simulator.hpp operator.hpp probe.hpp


# ********* nengo_mpi *************

nengo_mpi: nengo_mpi.o ${MPI_OBJS} | ${BIN}
	${MPICXX} -o ${BIN}/nengo_mpi nengo_mpi.o ${MPI_OBJS} ${DEFS} -std=${STD} ${BOOST_LIB} -lm ${HDF5_LIB} -lhdf5_openmpi -ldl  ${COMPRESSION_LIBS} -lcblas

nengo_mpi.o: nengo_mpi.cpp mpi_operator.hpp probe.hpp


# ********* mpi_sim.so *************

mpi_sim.so: ${MPI_OBJS} python.o | ${BIN}
	${MPICXX} -o ${BIN}/mpi_sim.so ${MPI_OBJS} python.o -shared ${DEFS} -std=${STD} ${BOOST_LIB} -lm ${PYTHON_LIB} -lboost_python ${HDF5_LIB} -lhdf5_openmpi -ldl ${COMPRESSION_LIBS} -lcblas

python.o: python.cpp python.hpp simulator.hpp chunk.hpp operator.hpp mpi_operator.hpp probe.hpp


# ********* common to all *************

mpi_operator.o: mpi_operator.cpp mpi_operator.hpp signal.hpp operator.hpp
mpi_simulator.o: mpi_simulator.cpp mpi_simulator.hpp simulator.hpp spec.hpp chunk.hpp psim_log.hpp
psim_log.o: psim_log.cpp psim_log.hpp sim_log.hpp spec.hpp

probe.o: probe.cpp probe.hpp signal.hpp
operator.o: operator.cpp operator.hpp signal.hpp
signal.o: signal.cpp signal.hpp
chunk.o: chunk.cpp chunk.hpp signal.hpp operator.hpp utils.hpp spec.hpp mpi_operator.hpp spaun.hpp probe.hpp sim_log.hpp psim_log.hpp
simulator.o: simulator.cpp simulator.hpp signal.hpp operator.hpp chunk.hpp spec.hpp
spec.o: spec.cpp spec.hpp
spaun.o: spaun.cpp spaun.hpp signal.hpp operator.hpp utils.hpp
sim_log.o: sim_log.cpp sim_log.hpp spec.hpp
utils.o: utils.cpp utils.hpp signal.hpp
debug.o: debug.cpp debug.hpp

${BIN}:
	mkdir ${BIN}
