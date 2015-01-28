#ifndef NENGO_MPI_MPI_SIM_HPP
#define NENGO_MPI_MPI_SIM_HPP

#include <list>
#include <string>

#include <mpi.h>

#include "chunk.hpp"

using namespace std;

const int stop_flag = -1;

const int add_signal_flag = 0;
const int add_op_flag = 1;
const int add_probe_flag = 2;

const int setup_tag = 1;
const int probe_tag = 2;

string recv_string(int src, int tag, MPI_Comm comm);
void send_string(string s, int dst, int tag, MPI_Comm comm);

float recv_float(int src, int tag, MPI_Comm comm);
void send_float(float f, int dst, int tag, MPI_Comm comm);

int recv_int(int src, int tag, MPI_Comm comm);
void send_int(int i, int dst, int tag, MPI_Comm comm);

key_type recv_key(int src, int tag, MPI_Comm comm);
void send_key(key_type i, int dst, int tag, MPI_Comm comm);

BaseMatrix* recv_matrix(int src, int tag, MPI_Comm comm);
void send_matrix(BaseMatrix* matrix, int dst, int tag, MPI_Comm comm);

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
    MPI_Comm comm;
    int num_remote_chunks;
    MpiSimulatorChunk* master_chunk;
};

#endif
