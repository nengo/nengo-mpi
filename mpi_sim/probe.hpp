
#ifndef NENGO_MPI_PROBE_HPP
#define NENGO_MPI_PROBE_HPP

#include <vector>
#include <memory>

#include "operator.hpp"

using namespace std;

class Probe {
public:
    Probe(SignalView signal, dtype period);
    void init_for_simulation(int n_steps, int fe);

    void gather(int n_steps);

    shared_ptr<dtype> flush_to_buffer(int &n_rows);

    // Gives up data currently stored in probe.
    // After this call, the probe will be empty.
    vector<unique_ptr<BaseSignal>> harvest_data();

    /* Makes sure the probe's buffer is empty. May be called multiple times in a single simulation. */
    void clear();

    /* Reset the probe. Only called between simulations. */
    void reset();

    string to_string() const;

    friend ostream& operator << (ostream &out, const Probe &probe){
        out << probe.to_string();
        return out;
    }

protected:
    // The data we've collected so far
    vector<unique_ptr<BaseSignal>> data;

    // The signal to record
    SignalView signal;

    // How frequently to sample the recorded signal
    dtype period;

    // The index of the next location to write to in the data vector.
    int data_index;

    // The current time index in the simulation.
    int time_index;

    int flush_every;

    shared_ptr<dtype> buffer;
};

#endif
