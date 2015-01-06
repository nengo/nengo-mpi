#ifndef NENGO_MPI_CHUNK_HPP
#define NENGO_MPI_CHUNK_HPP

#include <boost/serialization/map.hpp>
#include <boost/serialization/list.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

#include <map>
#include <list>
#include <string>
#include <sstream>
#include <vector>

#include "operator.hpp"
#include "mpi_operator.hpp"
#include "probe.hpp"
#include "debug.hpp"

// Class declaration for MpiSimulatorChunk

// The chunks contain a number of maps, and this typedef
// gives the type of the key used in those maps. When created
// from a python nengo simulation, the keys are typically
// addresses of python objects.
typedef unsigned long long int key_type;

/*
enum class OpType {
    RESET,
    COPY,
    DOT_INC,
    ELEMENTWISE_INC,
    LIF,
    LIF_RATE,
    RECTIFIED_LINEAR,
    SIGMOID,
    LINEAR_FILTER,
    MPI_SEND,
    MPI_RECV,
    MPI_WAIT
}

map<string, OpType> op_string_map {
    {"RESET": }};
    */

// An MpiSimulatorChunk represents the portion of a Nengo
// network that is simulated by a single MPI process.
class MpiSimulatorChunk{

public:
    MpiSimulatorChunk();
    MpiSimulatorChunk(string label, float dt);
    const string classname() { return "MpiSimulatorChunk"; }

    // Run an integer number of steps. Called by a
    // worker process once it gets a signal from the master
    // process telling the worker to begin a simulation
    void run_n_steps(int steps);

    // Add signals to the chunk. These contain the data
    // required by the simulation, as well as current simulation
    // state. The key must be unique, as its purpose is to allow
    // operators to reference the key.
    void add_signal(key_type key, string l, Matrix data);

    void add_probe(key_type probe_key, key_type signal, float period);
    void add_probe(key_type probe_key, Probe<Matrix>* probe);

    // Look up internal object by key
    Matrix* get_signal(key_type key);
    Matrix* get_signal(string key);
    Probe<Matrix>* get_probe(key_type key);

    // Functions used to add operators to the chunk. These
    // operators operate on signals that have been added
    // to the chunk in order to simulate the network. The
    // at the time an operator is added, all the signlas that
    // it operates on must have already been added to the chunk.
    // Note that the order that operators are added to the chunk
    // determines the order that they will be executed in.

    void add_mpi_send(MPISend* mpi_send);
    void add_mpi_recv(MPIRecv* mpi_recv);
    void add_mpi_wait(MPIWait* mpi_wait);

    void add_op(Operator* op);

    void add_op(string op_string);
    Matrix* extract_list(string s);

    // Called after the chunk is sent over to the MPI worker that
    // will simulate it. Makes the MPIWait operators collect the
    // boost.mpi request objects from their assigned MPISend or
    // MPIRecv operator. Thus can only be done after chunk is sent,
    // because the requests cannot be serialized
    void setup_mpi_waits();

    // Deprecated. This function isn't used anymore, but is kept around
    // in case it comes in handy. Finds the MPIWait operator that
    // is identified by a given tag/key. Was used to find the MPIWait
    // operator that is assigned to an MPISend or MPIRecv
    MPIWait* find_wait(int tag);

    double* get_time_pointer(){return &time;}
    int get_num_probes(){return probe_map.size();}

    string to_string() const;
    string print_maps();
    string print_signal_pointers();
    string print_signals();

    friend ostream& operator << (ostream &out, const MpiSimulatorChunk &chunk){
        out << chunk.to_string();
        return out;
    }

    map<int, MPISend*> mpi_sends;
    map<int, MPIRecv*> mpi_recvs;
    map<int, MPIWait*> mpi_waits;
    map<key_type, Probe<Matrix>*> probe_map;

    float dt;
    string label;

private:
    double time;
    int n_steps;
    map<key_type, Matrix*> signal_map;
    map<key_type, string> signal_labels;
    list<Operator*> operator_list;
    Operator* operators;
    int num_operators;

    // See http://www.boost.org/doc/libs/1_56_0/libs/serialization/doc/serialization.html
    // for info on how boost serialization works.
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version){
        dbg("Serializing: " << classname());

        // Make the serialization aware of derived classes.
        // Have to do this since we're serializing the operators through
        // a pointer whose type is Operator* (i.e. the base class).
        ar.template register_type<Reset>();
        ar.template register_type<Copy>();
        ar.template register_type<DotInc>();
        ar.template register_type<ElementwiseInc>();
        ar.template register_type<SimLIF>();
        ar.template register_type<SimLIFRate>();
        ar.template register_type<SimRectifiedLinear>();
        ar.template register_type<SimSigmoid>();
        ar.template register_type<Synapse>();
        ar.template register_type<MPISend>();
        ar.template register_type<MPIRecv>();
        ar.template register_type<MPIWait>();

        ar & label;
        ar & probe_map;
        ar & signal_map;
        ar & signal_labels;
        ar & operator_list;
        ar & time;
        ar & dt;
        ar & n_steps;

        ar & mpi_sends;
        ar & mpi_recvs;
        ar & mpi_waits;
    }
};

#endif
