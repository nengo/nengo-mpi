#ifndef NENGO_MPI_SIMULATOR_HPP
#define NENGO_MPI_SIMULATOR_HPP

#include <list>
#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>

#include "chunk.hpp"
#include "mpi_interface.hpp"
#include "debug.hpp"

using namespace std;
class MpiSimulator{
    static char delim;

public:
    MpiSimulator();
    MpiSimulator(int num_components, dtype dt, string out_filename);
    MpiSimulator(string in_filename, bool spawn);

    const string classname() { return "MpiSimulator"; }

    // After this is called, chunks cannot be added.
    // Must be called before run_n_steps
    void finalize();

    void run_n_steps(int steps, bool progress);
    vector<key_type> get_probe_keys();
    vector<BaseMatrix*> get_probe_data(key_type probe_key);

    void reset();

    void add_base_signal(int component, key_type key, string label, BaseMatrix* data);
    void add_op(int component, string op_string);
    void add_probe(int component, key_type probe_key, string signal_string, int period);

    string to_string() const;

    friend ostream& operator << (ostream &out, const MpiSimulator &sim){
        out << sim.to_string();
        return out;
    }

    int num_components;
    MpiSimulatorChunk* master_chunk;

private:
    bool spawn;
    MpiInterface mpi_interface;
    dtype dt;

    bool write_to_file;
    ofstream* out_file;

    // Place to store probe data retrieved from worker
    // processes after simulation has finished
    map<key_type, vector<BaseMatrix*> > probe_data;

    // Map from a source index to number of probes. Used for gather
    // probe data from remote chunks after simulation
    map<int, int> probe_counts;
};

#endif
