#ifndef NENGO_MPI_SIMULATOR_HPP
#define NENGO_MPI_SIMULATOR_HPP

#include <boost/serialization/list.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <list>
#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>

#include "chunk.hpp"
#include "mpi_simulator.hpp"
#include "debug.hpp"

using namespace std;
class MpiSimulator{

public:
    MpiSimulator();
    MpiSimulator(int num_components, float dt);

    const string classname() { return "MpiSimulator"; }

    // After this is called, chunks cannot be added.
    // Must be called before run_n_steps
    void finalize();

    void run_n_steps(int steps);
    vector<Matrix*>* get_probe_data(key_type probe_key);

    void add_signal(int component, key_type key, string label, Matrix* data);
    void add_op(int component, string op_string);
    void add_probe(int component, key_type probe_key, key_type signal_key, int period);

    void write_to_file(string filename);
    void read_from_file(string filename);

    string to_string() const;

    friend ostream& operator << (ostream &out, const MpiSimulator &sim){
        out << sim.to_string();
        return out;
    }

    int num_components;
    MpiSimulatorChunk* master_chunk;

private:
    MpiInterface mpi_interface;
    float dt;

    // Place to store probe data retrieved from worker
    // processes after simulation has finished
    map<key_type, vector<Matrix*>* > probe_data;

    // Map from a source index to number of probes
    map<int, int> probe_counts;

    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version){

        dbg("Serializing: " << classname());

        ar & master_chunk;
        ar & num_components;
        ar & dt;
    }
};

#endif
