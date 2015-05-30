#ifndef NENGO_MPI_MPI_SIMULATOR_HPP
#define NENGO_MPI_MPI_SIMULATOR_HPP

#include <list>
#include <vector>
#include <string>
#include <memory>
#include <exception>

#include <mpi.h>

#include "simulator.hpp"
#include "chunk.hpp"
#include "psim_log.hpp"

using namespace std;

const int stop_flag = -1;

const int add_signal_flag = 0;
const int add_op_flag = 1;
const int add_probe_flag = 2;

const int setup_tag = 1;
const int probe_tag = 2;

class MpiSimulator: public Simulator{
public:
    MpiSimulator(bool mpi_merged);

    // Used when we need to spaun extra process (e.g. when run through python)
    MpiSimulator(int n_processors, dtype dt, bool mpi_merged);

    ~MpiSimulator();

    void spawn_processors();
    void init();

    void add_base_signal(
        int component, key_type key, string label, unique_ptr<BaseSignal> data) override;
    void add_op(int component, string op_string) override;
    void add_probe(
        int component, key_type probe_key, string signal_string, dtype period, string name) override;

    SignalView get_signal(string signal_string) override;
    void add_op(unique_ptr<Operator> op) override;

    void finalize_build() override;

    void run_n_steps(int steps, bool progress, string log_filename) override;

    void gather_probe_data() override;

    string to_string() const;
    void from_file(string filename) override;

    friend ostream& operator << (ostream &out, const MpiSimulator &sim){
        out << sim.to_string();
        return out;
    }

protected:
    int n_processors;
    bool mpi_merged;

    MPI_Comm comm;

    // Map from a source index to number of probes. Used to gather
    // probe data from remote chunks after simulation.
    map<int, int> probe_counts;
};

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

#endif
