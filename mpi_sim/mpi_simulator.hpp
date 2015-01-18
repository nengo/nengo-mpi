#ifndef NENGO_MPI_MPI_SIM_HPP
#define NENGO_MPI_MPI_SIM_HPP

#include <list>
#include <string>

#include <mpi.h>
#include <boost/mpi.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/intercommunicator.hpp>

#include "flags.hpp"
#include "chunk.hpp"

namespace mpi = boost::mpi;
using namespace std;

class MpiInterface{
public:
    void initialize_chunks(bool spawn, MpiSimulatorChunk* chunk, int num_remote_chunks);

    void add_base_signal(int component, key_type key, string label, BaseMatrix* data);
    void add_op(int component, string op_string);
    void add_probe(int component, key_type probe_key, string signal_string, float period);
    void finalize();

    void run_n_steps(int steps, bool progress);
    void gather_probe_data(map<key_type, vector<BaseMatrix*> >& probe_data, map<int, int>& probe_counts);

    void finish_simulation();

private:
    mpi::communicator comm;
    int num_remote_chunks;
    MpiSimulatorChunk* master_chunk;
};

#endif
