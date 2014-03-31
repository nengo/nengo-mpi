
#include "operator.hpp"
#include "simulator.hpp"

Simulator::Simulator(Operator* operators):operators(operators){}

void Simulator::run_n_steps(int steps){
    Operator* o = operators;
    for(unsigned i = 0; i < num_operators; ++i){

        //Call the operator
        (*o)();

        //Move to next operator
        o++;
    }
}