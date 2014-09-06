#ifndef NENGO_MPI_SIMULATOR_HPP
#define NENGO_MPI_SIMULATOR_HPP

#include <boost/serialization/map.hpp>
#include <boost/serialization/list.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <map>
#include <list>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>

#include "operator.hpp"
#include "mpi_operator.hpp"
#include "probe.hpp"

using namespace std;

typedef unsigned long long int key_type;

class MpiSimulatorChunk{

public:
    MpiSimulatorChunk();
    MpiSimulatorChunk(double dt);
    const string classname() { return "MpiSimulatorChunk"; }

    void run_n_steps(int steps);

    void add_operator(Operator* op);
    void add_mpi_send(MPISend* mpi_send);
    void add_mpi_recv(MPIRecv* mpi_recv);
    void add_wait(MPIWait* mpi_wait);

    void add_probe(key_type key, Probe<Vector>* probe);

    void add_vector_signal(key_type key, Vector* sig);
    void add_matrix_signal(key_type key, Matrix* sig);
    MPIWait* find_wait(int tag);

    Probe<Vector>* get_probe(key_type key);
    Vector* get_vector_signal(key_type key);
    Matrix* get_matrix_signal(key_type key);

    double* get_time_pointer(){return &time;}

    string to_string() const;

    friend ostream& operator << (ostream &out, const MpiSimulatorChunk &chunk){
        out << chunk.to_string() << endl;
        return out;
    }

private:
    double time;
    double dt;
    int n_steps;
    map<key_type, Probe<Vector>*> probe_map;
    map<key_type, Matrix*> matrix_signal_map;
    map<key_type, Vector*> vector_signal_map;
    list<Operator*> operator_list;
    Operator* operators;
    int num_operators;
    list<MPIWait*> mpi_waits;

    // See http://www.boost.org/doc/libs/1_56_0/libs/serialization/doc/serialization.html
    // for how boost serialization works.
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version){

        dbg("Serializing: " << classname());

        // Make the serialization aware of derived classes.
        // Have to do this since we're serializing the operators through
        // a pointer whose type is Operator* (i.e. the base class).
        ar.template register_type<Reset>();
        ar.template register_type<Copy>();
        ar.template register_type<DotIncMV>();
        ar.template register_type<DotIncVV>();
        ar.template register_type<ProdUpdate>();
        ar.template register_type<SimLIF>();
        ar.template register_type<SimLIFRate>();
        ar.template register_type<MPISend>();
        ar.template register_type<MPIRecv>();
        ar.template register_type<MPIWait>();

        ar & probe_map;
        ar & matrix_signal_map;
        ar & vector_signal_map;
        ar & operator_list;
        ar & time;
        ar & dt;
        ar & n_steps;
    }
};

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
        out << sim.to_string() << endl;
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

//Forward declaration. Definition is in mpi_simulator.cpp.
void send_chunks(list<MpiSimulatorChunk*>);

#endif
