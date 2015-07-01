#ifndef NENGO_MPI_MPI_CHUNK_HPP
#define NENGO_MPI_MPI_CHUNK_HPP

#include <mpi.h>

#include "chunk.hpp"
#include "mpi_operator.hpp"
#include "mpi_sim_log.hpp"

/* An MpiSimulatorChunk represents the portion of a Nengo
 * network that is simulated by a single MPI process. */
class MpiSimulatorChunk: public SimulatorChunk{

public:
    MpiSimulatorChunk();
    MpiSimulatorChunk(int rank, int n_processors, MPI_Comm comm, bool mpi_merged);
    const string classname() { return "MpiSimulatorChunk"; }

    /* Add simulation objects to the chunk from an HDF5 file. */
    void from_file(string filename) override;

    /* Run an integer number of steps. Called by a
     * worker process once it gets a signal from the master
     * process telling the worker to begin a simulation. */
    void run_n_steps(int steps, bool progress) override;

    /* Add MPI-related operators. These have to be added separately,
     * because we need to initialize them in a special way before the
     * simulation begins. */
    void add_mpi_send(float index, int dst, int tag, SignalView content);
    void add_mpi_recv(float index, int src, int tag, SignalView content);

    void finalize_build();
    string to_string() override const;

    friend ostream& operator << (ostream &out, const MpiSimulatorChunk &chunk){
        out << chunk.to_string();
        return out;
    }

protected:
    int rank;
    MPI_Comm comm;

    list<unique_ptr<MPIOperator>> mpi_sends;
    list<unique_ptr<MPIOperator>> mpi_recvs;

    bool mpi_merged;

    // Used at build time to construct the merged mpi operators if mpi_merged is true
    map<int, vector<pair<int, SignalView>>> merged_sends;
    map<int, vector<pair<int, SignalView>>> merged_recvs;
    map<int, int> send_tags;
    map<int, int> recv_tags;

    // Iterators pointing to locations in operator_list
    // where the merged mpi ops should be added.
    map<int, float> send_indices;
    map<int, float> recv_indices;
};

#endif
