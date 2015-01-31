#ifndef NENGO_MPI_CHUNK_HPP
#define NENGO_MPI_CHUNK_HPP

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
#include "ezProgressBar-2.1.1/ezETAProgressBar.hpp"

// key type for various maps in the MpiSimulatorChunk. Keys are typically
// addresses of python objects, so we need to use long long ints (64 bits)
typedef long long int key_type;

// An MpiSimulatorChunk represents the portion of a Nengo
// network that is simulated by a single MPI process.
class MpiSimulatorChunk{

public:
    MpiSimulatorChunk();
    MpiSimulatorChunk(string label, dtype dt);
    const string classname() { return "MpiSimulatorChunk"; }

    // Run an integer number of steps. Called by a
    // worker process once it gets a signal from the master
    // process telling the worker to begin a simulation
    void run_n_steps(int steps, bool progress);

    // *** Signals ***

    // Add data to the chunk, in the form of a BaseSignal. These
    // contain the data required by the simulation, as well as
    // current simulation state. The key must be unique, as it
    // will later be used by operators to get a view of the BaseSignal.
    void add_base_signal(key_type key, string l, BaseMatrix* data);

    // Look up base signal by key.
    // Base signals are stored in C-format (Row-major).
    BaseMatrix* get_base_signal(key_type key);

    // Get a ``view'' on the base signal stored at the given key.
    //     shape1, shape2  : Shape of the returned view.
    //
    //     stride1, stide2 : Number of steps to take in the base signal
    //                       to get a new element of the view along that
    //                       dimension.
    //
    //     offset          : Index of the element in the base array that
    //                       the view begins at.
    Matrix get_signal(
        key_type key, int shape1, int shape2, int stride1, int stride2, int offset);

    // Get a ``view'' on a stored base signal from a String containing the key
    // of the base signal and parameters of the view.
    // Format of signal_string:
    //     key:(shape1, shape2):(stride1, stride2):offset
    //
    Matrix get_signal(string signal_string);

    // *** Operators ***

    // Functions used to add operators to the chunk. These
    // operators operate on signals that have been added
    // to the chunk in order to simulate the network.
    // At the time an operator is added, all the signlas that
    // it operates on must have already been added to the chunk.
    // Note that the order that operators are added to the chunk
    // determines the order that they will be executed in.
    void add_op(Operator* op);
    void add_op(string op_string);

    // Add MPI-related operators. These have to be added separately,
    // because we need to initiallize them in a special way before the
    // simulation begins.
    void add_mpi_send(MPISend* mpi_send);
    void add_mpi_recv(MPIRecv* mpi_recv);

    // *** Probes ***

    // Add a probe to the chunk. A probe can be used to record the state
    // of a signal as the simulation progresses.
    //     probe_key     : Unique key which can later be used to retrieve the probe.
    //
    //     signal_string : A string specifying a base signal and a view thereof.
    //                     Specifies what data the probe will record.
    //
    //     period        : How often the signal is sampled.
    void add_probe(key_type probe_key, string signal_string, dtype period);

    // Add a pre-created probe to the chunk.
    void add_probe(key_type probe_key, Probe* probe);

    // *** Miscellaneous ***

    // Helper function to extract a BaseMatrix from a string. Assumes
    // the data for the BaseMatrix is encoded in the string as a python list.
    static BaseMatrix* extract_list(string s);

    // Used to pass the time to python functions
    dtype* get_time_pointer(){return &time;}
    int get_num_probes(){return probe_map.size();}

    string to_string() const;

    friend ostream& operator << (ostream &out, const MpiSimulatorChunk &chunk){
        out << chunk.to_string();
        return out;
    }

    dtype dt;
    string label;

    vector<MPISend*> mpi_sends;
    vector<MPIRecv*> mpi_recvs;
    map<key_type, Probe*> probe_map;

private:
    dtype time;
    int n_steps;
    map<key_type, BaseMatrix*> signal_map;
    map<key_type, string> signal_labels;
    list<Operator*> operator_list;
};

#endif
