#pragma once

#include <map>
#include <list>
#include <string>
#include <sstream>
#include <vector>
#include <memory> // unique_ptr
#include <algorithm> // sort_stable
#include <utility> // pair
#include <exception>
#include <string>
#include <assert.h>

#include <mpi.h>

#include "signal.hpp"
#include "operator.hpp"
#include "utils.hpp"
#include "spec.hpp"
#include "mpi_operator.hpp"
#include "spaun.hpp"
#include "probe.hpp"
#include "sim_log.hpp"
#include "psim_log.hpp"
#include "ezProgressBar-2.1.1/ezETAProgressBar.hpp"

#include "typedef.hpp"
#include "debug.hpp"

// How frequently to flush the probe buffers, in units of number of steps.
const int FLUSH_PROBES_EVERY = 1000;

/* An MpiSimulatorChunk represents the portion of a Nengo
 * network that is simulated by a single MPI process. */
class MpiSimulatorChunk{

public:
    MpiSimulatorChunk(bool collect_timings);
    MpiSimulatorChunk(int rank, int n_processors, bool collect_timings);
    string classname() const { return "MpiSimulatorChunk"; }

    /* Add simulation objects to the chunk from an HDF5 file. */
    void from_file(string filename, hid_t file_plist, hid_t read_plist);

    /* Run an integer number of steps. Called by a
     * worker process once it gets a signal from the master
     * process telling the worker to begin a simulation. */
    void run_n_steps(int steps, bool progress);

    /* Reset the chunk. */
    void reset(unsigned seed);

    // *** Signals ***

    /* Add data to the chunk, in the form of a Signal. All data in
     * the simulation is stored in Signals, and Signals are an analog
     * of a Signal in the reference impl of nengo. The supplied key
     * must be unique, as it will later be used by operators to retrieve
     * views of the base Signal. */
    void add_base_signal(key_type key, Signal signal);

    /* Get a ``view'' of a base signal stored at the given key.
     * Most operators work in terms of these views.
     *
     *     key              : Key identifying the base signal that we
     *                       are getting a view of.
     *
     *     label            : Name for the view.
     *
     *     ndim             : Number of dimensions for returned view.
     *
     *     shape1, shape2   : Shape of the returned view.
     *
     *     stride1, stride2 : Number of steps to take in the base signal
     *                        to get a new element of the view along that
     *                        dimension.
     *
     *     offset           : Index of the element in the base array that
     *                        the view begins at.                         */
    Signal get_signal_view(
        key_type key, string label, unsigned ndim,
        unsigned shape1, unsigned shape2,
        int stride1, int stride2, unsigned offset);

    /* Get a ``view`` on a stored base signal from a SignalSpec object. */
    Signal get_signal_view(SignalSpec ss);

    /* Get a ``view`` on a stored base signal from a string
     * (by converting it into a SignalSpec first) . */
    Signal get_signal_view(string ss);

    /* Get a ``view`` on a stored base signal from a key. Parameters of the view
     * are derived from the signal itself (so the view will have the same shape
     * as the signal that it is a view of). */
    Signal get_signal(key_type key);

    // *** Operators ***

    /* Functions used to add operators to the chunk. These
     * operators access views of the base Signals stored in the chunk,
     * and operate on the data in those views to carry out the simulation.
     * At the time that an operator is added, all base Signals that it operates
     * on must have already been added to the chunk. Also note that the order
     * in which operators are added to the chunk determines the order they will
     * be executed in during the simulation. */

    /* Add an operator from an OpSpec object, which stores the type of operator
     * to add, as well as any parameters that operator needs (e.g. the Signals
     * that it operates on). Identifiies the type of operator that needs to
     * be created, and calls the constructor appropriately. */
    void add_op(OpSpec os);

    /* Add an existing operator to the chunk. */
    void add_op(float index, unique_ptr<Operator> op);

    /* Add a TimeUpdate operator. A chunk may have only one TimeUpdate operator,
     * so this function only has an effect the first time that it is called. */
    void add_time_update(float index, unique_ptr<TimeUpdate> op);

    /* Add MPI-related operators. These have to be added separately,
     * because we need to initialize them in a special way before the
     * simulation begins. */
    void add_mpi_send(float index, int dst, int tag, Signal content);
    void add_mpi_recv(float index, int src, int tag, Signal content, bool is_update);

    // *** Probes ***

    // Add a a probe from a ProbeSpec object.
    void add_probe(ProbeSpec ps);

    // *** Miscellaneous ***

    void finalize_build();
    void finalize_build(MPI_Comm comm);

    void set_log_filename(string lf);
    bool is_logging();
    void close_simulation_log();

    void flush_probes();
    size_t get_num_probes(){return probe_map.size();}

    void process_timing_data(
        int n_steps, const map<string, double>& per_class_average_timings,
        const double per_op_timings[], const vector<double>& step_times);

    string to_string() const;

    friend ostream& operator << (ostream &out, const MpiSimulatorChunk &chunk){
        out << chunk.to_string();
        return out;
    }

    dtype dt;
    string label;

    map<key_type, shared_ptr<Probe>> probe_map;
    vector<ProbeSpec> probe_info;

private:
    int rank;
    int n_processors;

    unique_ptr<SimulationLog> sim_log;
    string log_filename;

    map<key_type, Signal> signal_map;
    map<key_type, Signal> signal_init_value;

    // Contains all operators - don't have to worry about deleting these, since we
    // have unique_ptr's for all these ops in the lists below.
    list<Operator*> operator_list;

    // operate_store contains only non-mpi operators
    list<unique_ptr<Operator>> operator_store;

    list<unique_ptr<MPISend>> mpi_sends;
    list<unique_ptr<MPIRecv>> mpi_recvs;

    unique_ptr<TimeUpdate> time_update;

    bool collect_timings;
};

template <class A, class B> inline bool compare_first_lt(const pair<A, B> &left, const pair<A, B> &right){
    return (left.first < right.first);
}

template <class A, class B> inline bool compare_first_gt(const pair<A, B> &left, const pair<A, B> &right){
    return (left.first > right.first);
}

inline bool compare_indices(const OpSpec &left, const OpSpec &right){
    return (left.index < right.index);
}

inline bool compare_op_ptr(const Operator* left, const Operator* right){
    return (left->get_index() < right->get_index());
}
