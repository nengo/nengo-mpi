#pragma once

#include <boost/python.hpp>
#include <string>
#include <list>
#include <vector>
#include <iostream>

#include "simulator.hpp"
#include "mpi_simulator.hpp"
#include "operator.hpp"
#include "probe.hpp"
#include "spec.hpp"
#include "debug.hpp"

using namespace std;

namespace bpy = boost::python;
namespace bpyn = bpy::numeric;

void python_mpi_init();
void python_mpi_finalize();
int python_get_mpi_rank();
int python_get_mpi_n_procs();
void python_kill_workers();
void python_worker_start();

bool hasattr(bpy::object obj, string const &attrName);

unique_ptr<BaseSignal> ndarray_to_matrix(bpyn::array a);
unique_ptr<BaseSignal> list_to_matrix(bpy::list l);

/*
 * PythonMpiSimulator is a python-facing shell for MpiSimulator; it stores
 * an MpiSimulator, and most of its methods just call the corresponding
 * methods on that. Lets us keep all the code that requires python in one file. */
class PythonMpiSimulator{
public:
    PythonMpiSimulator();
    PythonMpiSimulator(bpy::object n_components);

    void load_network(bpy::object filename);
    void finalize_build();

    /* Methods for controlling simulation. */
    void run_n_steps(bpy::object steps, bpy::object progress, bpy::object log_filename);
    bpy::object get_probe_data(bpy::object probe_key, bpy::object make_array);

    void reset(bpy::object seed);
    void close();

    /* Methods for creating PyFunc operators. */
    void create_PyFunc(bpy::object py_fn, bpy::object t_in, bpy::object index);

    void create_PyFuncI(
        bpy::object py_fn, bpy::object t_in, bpy::object input,
        bpyn::array py_input, bpy::object index);

    void create_PyFuncO(
        bpy::object py_fn, bpy::object t_in,bpy::object output, bpy::object index);

    void create_PyFuncIO(
        bpy::object py_fn, bpy::object t_in, bpy::object input,
        bpyn::array py_input, bpy::object output, bpy::object index);

    string to_string() const;

private:
    unique_ptr<Simulator> sim;
};

class PyFunc: public Operator{
public:
    PyFunc(bpy::object py_fn, dtype* t_in);
    PyFunc(
        bpy::object py_fn, dtype* t_in, SignalView input, bpyn::array py_input);
    PyFunc(bpy::object py_fn, dtype* t_in, SignalView output);
    PyFunc(
        bpy::object py_fn, dtype* t_in, SignalView input,
        bpyn::array py_input, SignalView output);

    void operator()();
    virtual string to_string() const;

    // Used to initialize input and output when their values are not supplied.
    // The SignalView constructor requires a BaseSignal.
    static BaseSignal null_matrix;
    static ublas::slice null_slice;

private:
    SignalView input;
    SignalView output;

    dtype* time;

    bpy::object py_fn;
    bpyn::array py_input;
    //bpyn::array py_output;

    bool supply_time;
    bool supply_input;
    bool get_output;
};

BaseSignal PyFunc::null_matrix = BaseSignal(0,0);
ublas::slice PyFunc::null_slice = ublas::slice(0, 0, 0);
