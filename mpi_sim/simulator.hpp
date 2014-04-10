#ifndef NENGO_MPI_SIMULATOR_HPP
#define NENGO_MPI_SIMULATOR_HPP

#include <boost/serialization/list.hpp>

#include <map>
#include <list>

#include "probe.hpp"
#include "operator.hpp"

using namespace std;

typedef unsigned long long int key_type;

//TODO: add a probing system
class MpiSimulatorChunk{

    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version){

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
        ar.template register_type<MPIReceive>();

        ar & probe_map;
        ar & matrix_signal_map;
        ar & vector_signal_map;
        ar & operator_list;
        ar & time;
        ar & dt;
        ar & n_steps;
    }

public:
    MpiSimulatorChunk();
    MpiSimulatorChunk(double dt);

    void run_n_steps(int steps);

    void add_operator(Operator* op);
    
    void add_probe(key_type key, Probe<Vector>* probe);

    void add_vector_signal(key_type key, Vector* sig);
    void add_matrix_signal(key_type key, Matrix* sig);

    Probe<Vector>* get_probe(key_type key);
    Vector* get_vector_signal(key_type key);
    Matrix* get_matrix_signal(key_type key);

    double* get_time_pointer(){return &time;}

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
};

class MpiSimulator{
public:
    MpiSimulator(int, MpiSimulatorChunk*);
    void run_n_steps(int steps);

private:
    MpiSimulatorChunk* chunks;
    int num_chunks;
};

#endif
