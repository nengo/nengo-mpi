
#ifndef NENGO_MPI_PROBE_HPP
#define NENGO_MPI_PROBE_HPP

#include <vector>

#include "operator.hpp"

using namespace std;

class Probe {
public:
    Probe(SignalView signal, dtype period);
    void init_for_simulation(int n_steps);
    void gather(int n_steps);

    /* Gives up data currently stored in probe. After this call, the probe will be empty. */
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
    vector<unique_ptr<BaseSignal>> data;
    SignalView signal;
    dtype period;
    int index;
    int step_offset;
};

#endif
