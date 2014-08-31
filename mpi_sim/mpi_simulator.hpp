#ifndef NENGO_MPI_MPI_SIM_HPP
#define NENGO_MPI_MPI_SIM_HPP

#include <list>

#include <mpi.h>
#include <boost/mpi.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/intercommunicator.hpp>
#include <boost/serialization/string.hpp>

#include "simulator.hpp"

namespace mpi = boost::mpi;
using namespace std;

void send_chunks(list<MpiSimulatorChunk*> chunks);

#endif
