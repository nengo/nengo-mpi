
#include "operator.hpp"
#include "simulator.hpp"

MpiSimulatorChunk::MpiSimulatorChunk()
    :time(0.0), dt(0.001) {
}

MpiSimulatorChunk::MpiSimulatorChunk(double dt)
    :time(0.0), dt(dt) {
}

void MpiSimulatorChunk::run_n_steps(int steps){
    for(unsigned step = 0; step < steps; ++step){
        std::list<Operator*>::const_iterator it; 
        for(it = operator_list.begin(); it != operator_list.end(); ++it){
            //Call the operator
            (**it)();
        }

        time += dt;
    }
}

void MpiSimulatorChunk::add_operator(Operator *op){
    operator_list.push_back(op);
}


void MpiSimulatorChunk::add_vector_signal(key_type key, Vector* sig){
    vector_signal_map[key] = sig;
}

void MpiSimulatorChunk::add_matrix_signal(key_type key, Matrix* sig){
    matrix_signal_map[key] = sig;
}

Vector* MpiSimulatorChunk::get_vector_signal(key_type key){
    return vector_signal_map[key];
}

Matrix* MpiSimulatorChunk::get_matrix_signal(key_type key){
    return matrix_signal_map[key];
}

MpiSimulator::MpiSimulator(MpiSimulatorChunk* chunks){
}

void MpiSimulator::run_n_steps(int steps){
}
