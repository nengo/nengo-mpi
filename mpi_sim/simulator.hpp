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

    const string classname() { return "MpiSimulator"; }

    MpiSimulatorChunk* add_chunk();

    // After this is called, chunks cannot be added.
    // Must be called before run_n_steps
    void finalize();

    void run_n_steps(int steps);
    vector<Matrix*>* get_probe_data(key_type probe_key);

    void write_to_file(string filename);
    void read_from_file(string filename);

    string to_string() const;

    friend ostream& operator << (ostream &out, const MpiSimulator &sim){
        out << sim.to_string();
        return out;
    }

private:
    MpiSimulatorChunk* master_chunk;
    list<MpiSimulatorChunk*> remote_chunks;
    MpiInterface mpi_interface;

    map<key_type, vector<Matrix*>* > probe_data;

    // Map from a source index to number of probes
    map<int, int> probe_counts;

    int chunk_index;

    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version){

        dbg("Serializing: " << classname());

        ar & master_chunk;
        ar & remote_chunks;
    }
};

#endif
