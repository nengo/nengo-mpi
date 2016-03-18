#pragma once

#include <vector>
#include <memory>
#include <cmath>

#include "signal.hpp"

#include "typedef.hpp"
#include "debug.hpp"


using namespace std;

class Probe {
public:
    Probe(Signal signal, dtype period);
    void init_for_simulation(unsigned n_steps, unsigned flush_every_);

    void gather(unsigned n_steps);

    shared_ptr<dtype> flush_to_buffer(unsigned &n_rows);

    // Gives up data currently stored in probe.
    // After this call, the probe will be empty.
    vector<Signal> harvest_data();

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
    vector<Signal> data;

    // The signal to record
    Signal signal;
    bool signal_contiguous;

    // How frequently to sample the recorded signal
    dtype period;

    // The index of the next location to write to in the data vector.
    unsigned data_index;

    // The current time index in the simulation.
    unsigned time_index;

    unsigned flush_every;

    shared_ptr<dtype> buffer;
};
