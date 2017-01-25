#pragma once

#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_10_API_VERSION
#include <numpy/arrayobject.h>

#include <string>
#include <list>
#include <vector>
#include <iostream>

#include "signal.hpp"
#include "operator.hpp"
#include "simulator.hpp"
#include "mpi_simulator.hpp"

class PythonException: public runtime_error{
public:
    PythonException():runtime_error(""){};
    PythonException(const string& message):runtime_error(message){};
};

class PyFunc: public Operator{
public:
    PyFunc(
        PyObject* fn, Signal time, Signal input, Signal output,
        dtype* time_buffer, dtype* input_buffer, dtype* output_buffer);
    ~PyFunc();

    void operator()();
    virtual string to_string() const;

private:
    PyObject* fn;

    Signal time;
    Signal input;
    Signal output;

    dtype* time_buffer;
    dtype* input_buffer;
    dtype* output_buffer;
};
