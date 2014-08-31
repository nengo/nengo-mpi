#ifndef NENGO_MPI_PYTHON_HPP
#define NENGO_MPI_PYTHON_HPP

#include <boost/python.hpp>
#include <string>

#include "simulator.hpp"
#include "operator.hpp"
#include "mpi_operators.hpp"
#include "probe.hpp"

using namespace std;

namespace bpy = boost::python;
namespace bpyn = bpy::numeric;

bool is_vector(bpyn::array a);
Vector* ndarray_to_vector(bpyn::array a);
Matrix* ndarray_to_matrix(bpyn::array a);
Vector* list_to_vector(bpy::list l);
bool hasattr(bpy::object obj, string const &attrName);

class PythonMpiSimulatorChunk;

class PythonMpiSimulator{
public:
    PythonMpiSimulator();

    PythonMpiSimulatorChunk* add_chunk();

    void finalize();

    void run_n_steps(bpy::object steps);

private:
    list<PythonMpiSimulatorChunk*> py_chunks;
    MpiSimulator mpi_sim;
};

class PythonMpiSimulatorChunk{

    friend class PythonMpiSimulator;

public:
    PythonMpiSimulatorChunk();
    PythonMpiSimulatorChunk(MpiSimulatorChunk* mpi_sim_chunk);

    //TODO: factor this method out
    void run_n_steps(bpy::object steps);

    void add_vector_signal(bpy::object key, bpyn::array sig);

    void add_matrix_signal(bpy::object key, bpyn::array sig);

    bpy::object get_probe_data(bpy::object probe, bpy::object make_array);

    void create_Probe(bpy::object key,  bpy::object signal, bpy::object period);

    void create_Reset(bpy::object dst, bpy::object val);

    void create_Copy(bpy::object dst, bpy::object src);

    void create_DotIncMV(bpy::object A, bpy::object X, bpy::object Y);

    void create_DotIncVV(bpy::object A, bpy::object X, bpy::object Y);

    void create_ProdUpdate(bpy::object B, bpy::object Y);

    void create_Filter(bpy::object input, bpy::object output,
                       bpy::list numer, bpy::list denom);

    void create_SimLIF(bpy::object n_neurons, bpy::object tau_rc,
                       bpy::object tau_ref, bpy::object dt, bpy::object J,
                       bpy::object output);

    void create_SimLIFRate(bpy::object n_neurons, bpy::object tau_rc,
                           bpy::object tau_ref, bpy::object dt, bpy::object J,
                           bpy::object output);

    void create_MPISend(bpy::object dst, bpy::object tag, bpy::object content);
    void create_MPIRecv(bpy::object src, bpy::object tag, bpy::object content);
    void create_MPIWait(bpy::object content);

    void create_PyFunc(bpy::object py_fn, bpy::object t_in);
    void create_PyFuncO(bpy::object output, bpy::object py_fn, bpy::object t_in);
    void create_PyFuncI(bpy::object py_fn, bpy::object t_in,
                    bpy::object input, bpyn::array py_input);
    void create_PyFuncIO(bpy::object output, bpy::object py_fn, bpy::object t_in,
                    bpy::object input, bpyn::array py_input);

private:
    MpiSimulatorChunk* mpi_sim_chunk;
};

class PyFunc: public Operator{
public:
    PyFunc(Vector* output, bpy::object py_fn, double* t_in);
    PyFunc(Vector* output, bpy::object py_fn, double* t_in,
           Vector* input, bpyn::array py_input);
    void operator()();
    virtual void print(ostream &out) const;

private:
    Vector* output;
    Vector* input;

    double* time;

    bpy::object py_fn;
    bpyn::array py_input;
    //bpyn::array py_output;

    bool supply_time;
    bool supply_input;
};

#endif
