#pragma once

#include <list>
#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <exception>
#include <ctime>

#include "signal.hpp"
#include "operator.hpp"
#include "chunk.hpp"
#include "spec.hpp"

#include "typedef.hpp"
#include "debug.hpp"

using namespace std;

class Simulator{

public:
    Simulator(bool collect_timings);

    virtual ~Simulator(){};

    virtual void from_file(string filename);
    virtual void finalize_build();

    virtual Signal get_signal_view(string signal_string);
    virtual Signal get_signal(key_type key);
    virtual void add_pyfunc(float index, unique_ptr<Operator> pyfunc);

    virtual void run_n_steps(int steps, bool progress, string log_filename);

    virtual void gather_probe_data();
    vector<Signal> get_probe_data(key_type probe_key);

    virtual void reset(unsigned seed);
    virtual void close();

    virtual string to_string() const;

    friend ostream& operator << (ostream &out, const Simulator &sim){
        out << sim.to_string();
        return out;
    }

    dtype dt(){
        return chunk->dt;
    }

    virtual void write_to_time_file(char* filename, double delta);

    void write_to_loadtimes_file(double delta){
        char* filename = getenv("NENGO_MPI_LOADTIMES_FILE");
        write_to_time_file(filename, delta);
    }

    void write_to_runtimes_file(double delta){
        char* filename = getenv("NENGO_MPI_RUNTIMES_FILE");
        write_to_time_file(filename, delta);
    }

protected:
    unique_ptr<MpiSimulatorChunk> chunk;
    bool collect_timings;
    string label;

    // Place to store probe data retrieved from worker
    // processes after simulation has finished.
    map<key_type, vector<Signal>> probe_data;

    // Store the probe info so that we can scatter it to all
    // the other processes, which will allow all processes to
    // build the HDF5 output file correctly.
    vector<ProbeSpec> probe_info;
};
