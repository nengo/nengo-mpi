#ifndef NENGO_MPI_PYTHON_HPP
#define NENGO_MPI_PYTHON_HPP

#include <boost/python.hpp>
#include <string>
#include <list>
#include <vector>
#include <iostream>

#include "simulator.hpp"
#include "chunk.hpp"
#include "operator.hpp"
#include "mpi_operator.hpp"
#include "probe.hpp"
#include "debug.hpp"

using namespace std;

namespace bpy = boost::python;
namespace bpyn = bpy::numeric;

bool hasattr(bpy::object obj, string const &attrName);

BaseMatrix* ndarray_to_matrix(bpyn::array a);
BaseMatrix* list_to_matrix(bpy::list l);

class PythonMpiSimulator{
public:
    PythonMpiSimulator();
    PythonMpiSimulator(bpy::object num_components, bpy::object dt);
    void finalize();

    void run_n_steps(bpy::object steps);
    bpy::object get_probe_data(bpy::object probe_key, bpy::object make_array);

    void reset();

    void add_signal(bpy::object component, bpy::object key,
                    bpy::object label, bpyn::array data);

    void add_op(bpy::object component, bpy::object op_string);

    void add_probe(bpy::object component, bpy::object probe_key,  bpy::object signal_string, bpy::object period);

    void create_PyFunc(bpy::object py_fn, bpy::object t_in);
    void create_PyFuncI(bpy::object py_fn, bpy::object t_in,
                        bpy::object input, bpyn::array py_input);
    void create_PyFuncO(bpy::object py_fn, bpy::object t_in, bpy::object output);
    void create_PyFuncIO(bpy::object py_fn, bpy::object t_in,
                         bpy::object input, bpyn::array py_input, bpy::object output);

    string to_string() const;

private:
    MpiSimulator mpi_sim;
    MpiSimulatorChunk* master_chunk;
};

class PyFunc: public Operator{
public:
    PyFunc(bpy::object py_fn, double* t_in);
    PyFunc(bpy::object py_fn, double* t_in, Matrix input, bpyn::array py_input);
    PyFunc(bpy::object py_fn, double* t_in, Matrix output);
    PyFunc(bpy::object py_fn, double* t_in, Matrix input, bpyn::array py_input, Matrix output);

    void operator()();
    virtual string to_string() const;

    // Used to initialize input and output when their values are not supplied.
    // The Matrix constructor requires a BaseMatrix.
    static BaseMatrix null_matrix;
    static ublas::slice null_slice;

private:
    Matrix input;
    Matrix output;

    double* time;

    bpy::object py_fn;
    bpyn::array py_input;
    //bpyn::array py_output;

    bool supply_time;
    bool supply_input;
    bool get_output;
};

BaseMatrix PyFunc::null_matrix = BaseMatrix(0,0);
ublas::slice PyFunc::null_slice = ublas::slice(0, 0, 0);

#endif
