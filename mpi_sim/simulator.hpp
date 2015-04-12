#ifndef NENGO_SIMULATOR_HPP
#define NENGO_SIMULATOR_HPP

#include <list>
#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <exception>

#include "chunk.hpp"
#include "debug.hpp"

using namespace std;

class Simulator{

public:
    static char delim;

    Simulator();
    Simulator(dtype dt);
    virtual ~Simulator(){};

    const virtual string classname() { return "Simulator"; }

    virtual void add_base_signal(key_type key, string label, unique_ptr<BaseSignal> data);
    virtual void add_base_signal(int component, key_type key, string label, unique_ptr<BaseSignal> data);

    virtual void add_op(string op_string);
    virtual void add_op(int component, string op_string);

    virtual void add_probe(key_type probe_key, string signal_string, dtype period, string name);
    virtual void add_probe(
        int component, key_type probe_key, string signal_string, dtype period, string name);

    virtual SignalView get_signal(string signal_string);
    virtual void add_op(unique_ptr<Operator> op);

    virtual void finalize_build();

    virtual void run_n_steps(int steps, bool progress, string log_filename);

    virtual void gather_probe_data();
    virtual vector<unique_ptr<BaseSignal>> get_probe_data(key_type probe_key);
    vector<key_type> get_probe_keys();

    virtual void reset();

    virtual string to_string() const;

    friend ostream& operator << (ostream &out, const Simulator &sim){
        out << sim.to_string();
        return out;
    }

    dtype* get_time_pointer(){
        return chunk->get_time_pointer();
    }

    dtype dt;

protected:
    shared_ptr<MpiSimulatorChunk> chunk;

    // Place to store probe data retrieved from worker
    // processes after simulation has finished.
    map<key_type, vector<unique_ptr<BaseSignal>>> probe_data;

    // Store the probe info so that we can scatter it to all
    // the other processes, which will allow all processes to
    // build the HDF5 output file correctly.
    vector<string> probe_info;
};

#endif
