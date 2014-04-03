#ifndef NENGO_MPI_SIMULATOR_HPP
#define NENGO_MPI_SIMULATOR_HPP

#include <map>
#include <list>


#include "operator.hpp"

typedef int key_type;

//TODO: add a probing system
class MpiSimulatorChunk{
public:
    MpiSimulatorChunk();
    MpiSimulatorChunk(double dt);

    void run_n_steps(int steps);

    void add_operator(Operator* op);

    void add_vector_signal(key_type key, Vector* sig);
    void add_matrix_signal(key_type key, Matrix* sig);

    Vector* get_vector_signal(key_type key);
    Matrix* get_matrix_signal(key_type key);

    double* get_time_pointer(){return &time;}

private:
    double time;
    double dt;
    std::map<key_type, Matrix*> matrix_signal_map;
    std::map<key_type, Vector*> vector_signal_map;
    std::list<Operator*> operator_list;
    Operator* operators;
    int num_operators;
};

class MpiSimulator{
public:
    MpiSimulator(MpiSimulatorChunk* chunks);
    void run_n_steps(int steps);

private:
    MpiSimulatorChunk* chunks;
    int num_chunks;
};

#endif
