#ifndef NENGO_MPI_CHUNK_HPP
#define NENGO_MPI_CHUNK_HPP

#include <map>
#include <list>
#include <string>
#include <sstream>
#include <vector>
#include <memory>
#include <exception>
#include <string>
#include <mpi.h>

#include "operator.hpp"
#include "mpi_operator.hpp"
#include "custom_ops.hpp"
#include "probe.hpp"
#include "sim_log.hpp"
#include "debug.hpp"
#include "ezProgressBar-2.1.1/ezETAProgressBar.hpp"

/* An MpiSimulatorChunk represents the portion of a Nengo
 * network that is simulated by a single MPI process. */
class MpiSimulatorChunk{

public:
    MpiSimulatorChunk();
    MpiSimulatorChunk(int component, string label, dtype dt);
    const string classname() { return "MpiSimulatorChunk"; }

    /* Run an integer number of steps. Called by a
     * worker process once it gets a signal from the master
     * process telling the worker to begin a simulation. */
    void run_n_steps(int steps, bool progress);

    // *** Signals ***

    /* Add data to the chunk, in the form of a BaseSignal. All data
     * in the simulation is stored in BaseSignals, and BaseSignals are an
     * analog of a Signal (not but not a SignalView) in the reference impl.
     * The supplied key must be unique, as it will later be used by operators
     * to retrieve a view of the BaseSignal. */
    void add_base_signal(key_type key, string l, unique_ptr<BaseSignal> signal);

    /* Get a ``view'' on the BaseSignal stored at the given key.
     * Most operators work in terms of these views.
     *
     *     shape1, shape2  : Shape of the returned view.
     *
     *     stride1, stide2 : Number of steps to take in the base signal
     *                       to get a new element of the view along that
     *                       dimension.
     *
     *     offset          : Index of the element in the base array that
     *                       the view begins at.                         */
    SignalView get_signal_view(
        key_type key, int shape1, int shape2, int stride1, int stride2, int offset);

    /* Get a ``view'' on a stored base signal from a string containing the key
     * of the base signal and parameters of the view.
     * Expected format of signal_string:
     *     key:(shape1, shape2):(stride1, stride2):offset */
    SignalView get_signal_view(string signal_string);

    /* Get a ``view'' on a stored base signal from a key. Parameters of the view
     * are derived from the signal itself (so the view will have the same shape
     * as the signal that it is a view of). */
    SignalView get_signal_view(key_type key);

    // *** Operators ***

    /* Functions used to add operators to the chunk. These
     * operators access views of the BaseMatrices stored in the chunk,
     * and operate on the data in those views to carry out the simulation.
     * At the time an operator is added, all data that it operates on must
     * have already been added to the chunk. Also note that the order in which
     * operators are added to the chunk determines the order they will
     * be executed in at simulation time. */
    void add_op(unique_ptr<Operator> op);
    void add_op(string op_string);

    /* Add MPI-related operators. These have to be added separately,
     * because we need to initialize them in a special way before the
     * simulation begins. */
    void add_mpi_send(unique_ptr<MPISend> mpi_send);
    void add_mpi_recv(unique_ptr<MPIRecv> mpi_recv);

    // *** Probes ***

    /* Add a probe to the chunk. A probe can be used to record the state
     * of a signal as the simulation progresses.
     *
     *     probe_key     : Unique key which can later be used to retrieve the probe.
     *
     *     signal_string : A string specifying a base signal and a view thereof.
     *                     Specifies what data the probe will record.
     *
     *     period        : How often the signal is sampled. */
    void add_probe(key_type probe_key, string signal_string, dtype period);

    // Add a pre-created probe to the chunk.
    void add_probe(key_type probe_key, shared_ptr<Probe> probe);

    // *** Miscellaneous ***

    // Set the simulation log object which records the results of the simulation
    void set_simulation_log(SimulationLog sl);
    void set_log_filename(string lf);
    void flush_probes();

    // Set the communicator used by the mpi ops
    void set_communicator(MPI_Comm comm);

    // Used to pass the simulation time to python functions
    dtype* get_time_pointer(){return &time;}

    int get_num_probes(){return probe_map.size();}

    string to_string() const;

    friend ostream& operator << (ostream &out, const MpiSimulatorChunk &chunk){
        out << chunk.to_string();
        return out;
    }

    dtype dt;
    string label;

    map<key_type, shared_ptr<Probe>> probe_map;

private:
    dtype time;
    int n_steps;
    int component;

    SimulationLog sim_log;
    string log_filename;

    map<key_type, string> signal_labels;
    map<key_type, shared_ptr<BaseSignal>> signal_map;

    // Contains all operators
    list<Operator*> operator_list;

    // Contains non-mpi operators
    list<unique_ptr<Operator>> operator_store;

    list<unique_ptr<MPISend>> mpi_sends;
    list<unique_ptr<MPIRecv>> mpi_recvs;
};

#endif
