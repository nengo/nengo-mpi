#pragma once

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

extern "C" {

/* Methods for controlling workers. See nengo_mpi/__main__.py. */
void init();
void finalize();
int get_rank();
int get_n_procs();
void kill_workers();
void worker_start();

typedef void (*py_func_t)();

/* Methods for building simulator. */
void create_simulator();
void load_network(char* filename);
void finalize_build();

/* Methods for running simulator. */
void run_n_steps(int n_steps, int progress, char* log_filename);
dtype* get_probe_data(key_type probe_key, size_t* n_signals, size_t* signal_size);
dtype* get_signal_value(key_type key, size_t* shape1, size_t* shape2);
void free_ptr(dtype* ptr);
void reset_simulator(unsigned seed);
void close_simulator();

void create_PyFunc(
        py_func_t py_fn, char* time_string, char* input_string, char* output_string,
        dtype* py_time, dtype* py_input, dtype* py_output, float index);

} // end extern "C"

class PyFunc: public Operator{
public:
    PyFunc(
        py_func_t py_fn, Signal time, Signal input, Signal output,
        dtype* py_time, dtype* py_input, dtype* py_output);

    void operator()();
    virtual string to_string() const;

private:
    py_func_t py_fn;

    Signal time;
    Signal input;
    Signal output;

    dtype* py_time;
    dtype* py_input;
    dtype* py_output;
};
