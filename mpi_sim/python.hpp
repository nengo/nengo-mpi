#pragma once

#include <boost/python.hpp>
#include <string>
#include <list>
#include <vector>
#include <iostream>

#include "signal.hpp"
#include "operator.hpp"
#include "simulator.hpp"
#include "mpi_simulator.hpp"

#include "typedef.hpp"
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

Signal ndarray_to_matrix(bpyn::array a);
Signal list_to_matrix(bpy::list l);

/*
 * PythonMpiSimulator is a python-facing wrapper for MpiSimulator; it stores
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
    bpy::object get_signal_value(bpy::object key, bpy::object make_array);

    void reset(bpy::object seed);
    void close();

    void create_PyFunc(
        bpy::object py_fn, bpy::object t, bpy::object input,
        bpyn::array py_input, bpy::object output, bpy::object index);

    string to_string() const;

private:
    unique_ptr<Simulator> sim;
};

class PyFunc: public Operator{
public:
    PyFunc(
        bpy::object py_fn, Signal t, Signal input,
        bpyn::array py_input, Signal output);

    void operator()();
    virtual string to_string() const;

private:
    Signal t;
    Signal input;
    Signal output;

    bpy::object py_fn;
    bpyn::array py_input;
};
