#ifndef NENGO_MPI_MPI_PYTHON_HPP
#define NENGO_MPI_MPI_PYTHON_HPP

#include "python.hpp"
#include "mpi_simulator.hpp"

/*
 * NativeSimulator is a python-facing shell for an MpiSimulator; most
 * of its methods just call the corresponding methods on the MpiSimulator
 * that it wraps. */
class NativeMpiSimulator: public NativeSimulator{
public:
    NativeMpiSimulator(){};
    NativeMpiSimulator(bpy::object num_components, bpy::object dt);
};

#endif
