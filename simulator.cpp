#include "simulator.hpp"
#include "operator.hpp"

Simulator::Simulator(Operator* operators):operators(operators){}

Simulator::run_n_steps(int steps){
    Operator* o = operators;
    for(unsigned i = 0; i < num_operators; ++i){

        //Call the operator
        (*o)();

        //Move to next operator
        o++;
    }
}