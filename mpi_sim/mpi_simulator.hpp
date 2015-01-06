#ifndef NENGO_MPI_MPI_SIM_HPP
#define NENGO_MPI_MPI_SIM_HPP

#include <list>
#include <string>

#include <mpi.h>
#include <boost/mpi.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/intercommunicator.hpp>
#include <boost/serialization/string.hpp>

#include "flags.hpp"
#include "chunk.hpp"

namespace mpi = boost::mpi;
using namespace std;

class MpiInterface{
public:
    void initialize_chunks(MpiSimulatorChunk* chunk, int num_remote_chunks);

    void add_signal(int component, key_type key, string label, Matrix* data);
    void add_op(int component, string op_string);
    void add_probe(int component, key_type probe_key, key_type signal_key, float period);
    void finalize();

    void run_n_steps(int steps);
    void gather_probe_data(map<key_type, vector<Matrix*>* >& probe_data, map<int, int>& probe_counts);

    void finish_simulation();

private:
    mpi::communicator comm;
    int num_remote_chunks;
    MpiSimulatorChunk* master_chunk;
};

#endif
