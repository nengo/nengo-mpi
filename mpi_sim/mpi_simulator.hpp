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

#include "typedef.hpp"
#include "debug.hpp"


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
int mpi_get_rank();
int mpi_get_n_procs();
void mpi_wake_workers();
void mpi_kill_workers();
void mpi_worker_start();
void mpi_worker_start(MPI_Comm comm);

string recv_string(int src, int tag, MPI_Comm comm);
void send_string(string s, int dst, int tag, MPI_Comm comm);

dtype recv_dtype(int src, int tag, MPI_Comm comm);
void send_dtype(dtype d, int dst, int tag, MPI_Comm comm);

int recv_int(int src, int tag, MPI_Comm comm);
void send_int(int i, int dst, int tag, MPI_Comm comm);

unsigned recv_unsigned(int src, int tag, MPI_Comm comm);
void send_unsigned(unsigned i, int dst, int tag, MPI_Comm comm);

key_type recv_key(int src, int tag, MPI_Comm comm);
void send_key(key_type i, int dst, int tag, MPI_Comm comm);

Signal recv_base_signal(int src, int tag, MPI_Comm comm);
void send_base_signal(Signal signal, int dst, int tag, MPI_Comm comm);

int bcast_recv_int(MPI_Comm comm);
void bcast_send_int(int i, MPI_Comm comm);

unsigned bcast_recv_unsigned(MPI_Comm comm);
void bcast_send_unsigned(unsigned i, MPI_Comm comm);
