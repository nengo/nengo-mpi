#include "operator.hpp"

//TODO: add a probing system
class Simulator{
public:
    Simulator(Operator* operators);
    run_n_steps(int steps);

private:
    Operator* operators;
    int num_operators;
};
