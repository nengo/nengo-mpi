#ifndef NENGO_MPI_PYTHON_HPP
#define NENGO_MPI_PYTHON_HPP

#include <boost/python.hpp>

#include "simulator.hpp"
#include "operator.hpp"

using namespace std;

namespace bpy = boost::python;
namespace bpyn = bpy::numeric;

bool is_vector(bpyn::array a);
Vector* ndarray_to_vector(bpyn::array a);
Matrix* ndarray_to_matrix(bpyn::array a);

class PythonMpiSimulatorChunk{
public:
    void run_n_steps(bpy::object steps);

    void add_signal(bpy::object key, bpyn::array sig);

    void create_Reset(bpy::object dst, bpy::object val);

    void create_Copy(bpy::object dst, bpy::object src);

    void create_DotInc(bpy::object A, bpy::object X, bpy::object Y);

    void create_ProdUpdate(bpy::object A, bpy::object X, bpy::object B, bpy::object Y);

    void create_SimLIF(bpy::object n_neurons, bpy::object tau_rc, 
                    bpy::object tau_ref, bpy::object dt, bpy::object J, bpy::object output);

    void create_SimLIFRate(bpy::object n_neurons, bpy::object tau_rc, 
                    bpy::object tau_ref, bpy::object dt, bpy::object J, bpy::object output);

    void create_MPISend();

    void create_MPIReceive();

    void create_PyFunc(bpy::object output, bpy::object py_fn, bpy::object t_in);

    void create_PyFuncWithInput(bpy::object output, bpy::object py_fn, 
                    bpy::object t_in, bpy::object input, bpyn::array py_input);

private:
    MpiSimulatorChunk mpi_sim_chunk;
};

class PyFunc: public Operator{
public:
    PyFunc(Vector* output, bpy::object py_fn, bool t_in);
    PyFunc(Vector* output, bpy::object py_fn, bool t_in, Vector* input, bpyn::array py_input);
    void operator()();
    friend ostream& operator << (ostream &out, const PyFunc &py_func);

private:
    Vector* output;
    Vector* input;

    bpy::object py_fn;
    bpyn::array py_input;
    //bpyn::array py_output;

    bool supply_time;
    bool supply_input;
};

#endif
