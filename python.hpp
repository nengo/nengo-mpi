#ifndef NENGO_MPI_PYTHON_HPP
#define NENGO_MPI_PYTHON_HPP

#include <boost/python.hpp>

#include "simulator.hpp"
#include "operator.hpp"

namespace bpy = boost::python;
namespace bpyn = bpy::numeric;

Vector* ndarray_to_vector(bpyn::array a);

class PythonMpiSimulatorChunk{
public:
    void add_signal(bpy::object key, bpyn::array sig);
    void create_Reset(bpy::object dst, bpy::object value);
    void create_Copy(bpy::object dst, bpy::object src);
    void create_DotInc(bpy::object A, bpy::object X, bpy::object Y);
    void create_ProdUpdate(bpy::object A, bpy::object X, bpy::object B, bpy::object Y);
    void create_SimLIF(bpy::object n_neuron, bpy::object tau_rc, bpy::object tau_ref, bpy::object dt, bpy::object J, bpy::object output);
    void create_SimLIFRate(bpy::object n_neurons, bpy::object tau_rc, bpy::object tau_ref, bpy::object dt, bpy::object J, bpy::object output);
    void create_MPISend();
    void create_MPIReceive();

private:
    MpiSimulatorChunk mpi_sim_chunk;
};

#endif
