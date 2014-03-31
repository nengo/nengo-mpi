#ifndef NENGO_MPI_SIMULATOR_HPP
#define NENGO_MPI_SIMULATOR_HPP

#include "operator.hpp"

//TODO: add a probing system
class Simulator{
public:
    Simulator(Operator* operators);
    void run_n_steps(int steps);

private:
    Operator* operators;
    int num_operators;
};

#endif
