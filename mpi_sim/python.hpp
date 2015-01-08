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

Matrix* ndarray_to_matrix(bpyn::array a);
Matrix* list_to_matrix(bpy::list l);

class PythonMpiSimulator{
public:
    PythonMpiSimulator();
    PythonMpiSimulator(bpy::object num_components, bpy::object dt);
    void finalize();

    void run_n_steps(bpy::object steps);
    bpy::object get_probe_data(bpy::object probe_key, bpy::object make_array);

    void reset();

    void write_to_file(string filename);
    void read_from_file(string filename);

    void add_signal(bpy::object component, bpy::object key,
                    bpy::object label, bpyn::array data);

    void add_op(bpy::object component, bpy::object op_string);

    void add_probe(bpy::object component, bpy::object probe_key,  bpy::object signal_key, bpy::object period);

    void create_PyFunc(bpy::object py_fn, bpy::object t_in);
    void create_PyFuncO(bpy::object output, bpy::object py_fn, bpy::object t_in);
    void create_PyFuncI(bpy::object py_fn, bpy::object t_in,
                    bpy::object input, bpyn::array py_input);
    void create_PyFuncIO(bpy::object output, bpy::object py_fn, bpy::object t_in,
                    bpy::object input, bpyn::array py_input);

    string to_string() const;

private:
    MpiSimulator mpi_sim;
    MpiSimulatorChunk* master_chunk;
};

class PyFunc: public Operator{
public:
    PyFunc(Matrix* output, bpy::object py_fn, double* t_in);
    PyFunc(Matrix* output, bpy::object py_fn, double* t_in,
           Matrix* input, bpyn::array py_input);

    void operator()();
    virtual string to_string() const;

private:
    Matrix* output;
    Matrix* input;

    double* time;

    bpy::object py_fn;
    bpyn::array py_input;
    //bpyn::array py_output;

    bool supply_time;
    bool supply_input;

    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version){
        //TODO
        ar & input;
        ar & output;
        ar & time;
    }
};

#endif
