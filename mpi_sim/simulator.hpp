#ifndef NENGO_MPI_SIMULATOR_HPP
#define NENGO_MPI_SIMULATOR_HPP

#include <list>
#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <exception>

#include "chunk.hpp"
#include "mpi_interface.hpp"
#include "debug.hpp"

using namespace std;

class MpiSimulator{
    static char delim;

public:
    MpiSimulator();
    MpiSimulator(int num_components, dtype dt);
    MpiSimulator(string in_filename, bool spawn);

    const string classname() { return "MpiSimulator"; }

    // After this is called, neither signals nor ops may be added.
    // Must be called before run_n_steps.
    void finalize();

    void run_n_steps(int steps, bool progress, string log_filename);
    vector<key_type> get_probe_keys();
    vector<unique_ptr<BaseSignal>> get_probe_data(key_type probe_key);

    void reset();

    void add_base_signal(
        int component, key_type key, string label,unique_ptr<BaseSignal> data);

    SignalView get_signal_from_master(string signal_string);

    void add_op(int component, string op_string);
    void add_op_to_master(unique_ptr<Operator> op);

    void add_probe(
        int component, key_type probe_key, string signal_string, int period);

    string to_string() const;

    friend ostream& operator << (ostream &out, const MpiSimulator &sim){
        out << sim.to_string();
        return out;
    }

    dtype* get_time_pointer(){
        return master_chunk->get_time_pointer();
    }

    dtype dt;

private:
    int num_components;
    shared_ptr<MpiSimulatorChunk> master_chunk;

    bool spawn;
    MpiInterface mpi_interface;

    // Place to store probe data retrieved from worker
    // processes after simulation has finished.
    map<key_type, vector<unique_ptr<BaseSignal>>> probe_data;

    // Map from a source index to number of probes. Used to gather
    // probe data from remote chunks after simulation.
    map<int, int> probe_counts;

    // Store the probe info so that we can scatter it to all
    // the other processes, which will allow all processes to
    // build the HDF5 output file correctly.
    vector<string> probe_info;
};

#endif
