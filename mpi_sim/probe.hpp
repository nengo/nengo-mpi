
#ifndef NENGO_MPI_PROBE_HPP
#define NENGO_MPI_PROBE_HPP

#include <vector>

#include "operator.hpp"

using namespace std;

class Probe {
public:
    Probe(Matrix signal, float period);
    void init_for_simulation(int n_steps);
    void gather(int n_steps);
    vector<BaseMatrix*> get_data();
    void clear(bool del);
    void reset();
    string to_string() const;

    friend ostream& operator << (ostream &out, const Probe &probe){
        out << probe.to_string();
        return out;
    }

protected:
    vector<BaseMatrix*> data;
    Matrix signal;
    float period;
    int index;
    int step_offset;
};

#endif
