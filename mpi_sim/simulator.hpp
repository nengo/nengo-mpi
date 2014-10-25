#ifndef NENGO_MPI_SIMULATOR_HPP
#define NENGO_MPI_SIMULATOR_HPP

#include <boost/serialization/list.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <list>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>

#include "chunk.hpp"
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

    void write_to_file(string filename);
    void read_from_file(string filename);

    string to_string() const;

    friend ostream& operator << (ostream &out, const MpiSimulator &sim){
        out << sim.to_string();
        return out;
    }

private:
    list<MpiSimulatorChunk*> chunks;

    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version){

        dbg("Serializing: " << classname());

        ar & chunks;
    }
};

// Forward declaration. Definition is in mpi_simulator.cpp.
// Avoids our having to include mpi_simulator.hpp, which would result
// in a circular include.
void send_chunks(list<MpiSimulatorChunk*>);

#endif
