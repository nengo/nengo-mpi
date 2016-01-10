#pragma once

#include <list>
#include <vector>
#include <string>
#include <memory>
#include <exception>

#include <mpi.h>

#include "simulator.hpp"
#include "spec.hpp"
#include "chunk.hpp"
#include "psim_log.hpp"

using namespace std;

const int setup_tag = 1;
const int probe_tag = 2;

extern int n_processors_available;

class MpiSimulator: public Simulator{
public:
    MpiSimulator(bool mpi_merged, bool collect_timings);
    ~MpiSimulator();

    void from_file(string filename) override;
    void finalize_build() override;

    void run_n_steps(int steps, bool progress, string log_filename) override;

    void gather_probe_data() override;

    void reset(unsigned seed) override;
    void close() override;

    string to_string() const;

    friend ostream& operator << (ostream &out, const MpiSimulator &sim){
        out << sim.to_string();
        return out;
    }

    void write_to_time_file(char* filename, double delta) override;

protected:
    int n_processors;
    bool mpi_merged;

    MPI_Comm comm;

    // Used to gather probe data from workers after simulation.
    vector<int> probe_counts;
};

void mpi_init();
void mpi_finalize();
int get_mpi_rank();
int get_mpi_n_procs();
void wake_workers();
void kill_workers();
void worker_start();
void worker_start(MPI_Comm comm);

string recv_string(int src, int tag, MPI_Comm comm);
void send_string(string s, int dst, int tag, MPI_Comm comm);

dtype recv_dtype(int src, int tag, MPI_Comm comm);
void send_dtype(dtype d, int dst, int tag, MPI_Comm comm);

int recv_int(int src, int tag, MPI_Comm comm);
void send_int(int i, int dst, int tag, MPI_Comm comm);

key_type recv_key(int src, int tag, MPI_Comm comm);
void send_key(key_type i, int dst, int tag, MPI_Comm comm);

unique_ptr<BaseSignal> recv_matrix(int src, int tag, MPI_Comm comm);
void send_matrix(unique_ptr<BaseSignal> matrix, int dst, int tag, MPI_Comm comm);

int bcast_recv_int(MPI_Comm comm);
void bcast_send_int(int i, MPI_Comm comm);
