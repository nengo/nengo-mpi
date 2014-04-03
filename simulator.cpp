
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
    try{
        Vector* vec = vector_signal_map.at(key);
        return vec;
    }catch(const out_of_range& e){
        cerr << "Error accessing MpiSimulatorChunk :: vector signal with key " << key << endl;
        throw e;
    }
}

Matrix* MpiSimulatorChunk::get_matrix_signal(key_type key){
    try{
        Matrix* mat = matrix_signal_map.at(key);
        return mat;
    }catch(const out_of_range& e){
        cerr << "Error accessing MpiSimulatorChunk matrix signal with key " << key << endl;
        throw e;
    }
}

MpiSimulator::MpiSimulator(MpiSimulatorChunk* chunks){
}

void MpiSimulator::run_n_steps(int steps){
}
