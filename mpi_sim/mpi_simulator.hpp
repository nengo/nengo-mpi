#ifndef NENGO_MPI_MPI_SIM_HPP
#define NENGO_MPI_MPI_SIM_HPP

#include <list>

#include <mpi.h>
#include <boost/mpi.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/intercommunicator.hpp>
#include <boost/serialization/string.hpp>

#include "chunk.hpp"

namespace mpi = boost::mpi;
using namespace std;

class MpiInterface{
public:
    void initialize_chunks(MpiSimulatorChunk* chunk, list<MpiSimulatorChunk*> remote_chunks);
    void start_simulation(int steps);
private:
    mpi::communicator comm;
    MpiSimulatorChunk* master_chunk;
};

#endif
