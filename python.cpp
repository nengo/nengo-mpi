#include <boost/python.hpp>
#include "python.hpp"
#include "simulator.hpp"
#include "operator.hpp"

namespace bpy = boost::python;

void PythonMpiSimulatorChunk::add_signal(bpy::object key, bpy::object sig){}
void PythonMpiSimulatorChunk::create_Reset(bpy::object dst, bpy::object value){}
void PythonMpiSimulatorChunk::create_Copy(bpy::object dst, bpy::object src){}
void PythonMpiSimulatorChunk::create_DotInc(bpy::object A, bpy::object X, bpy::object Y){}
void PythonMpiSimulatorChunk::create_ProdUpdate(bpy::object A, bpy::object X, bpy::object B, bpy::object Y){}
void PythonMpiSimulatorChunk::create_SimLIF(bpy::object n_neuron, bpy::object tau_rc, bpy::object tau_ref, bpy::object dt, bpy::object J, bpy::object output){}
void PythonMpiSimulatorChunk::create_SimLIFRate(bpy::object n_neurons, bpy::object tau_rc, bpy::object tau_ref, bpy::object dt, bpy::object J, bpy::object output){}
void PythonMpiSimulatorChunk::create_MPISend(){}
void PythonMpiSimulatorChunk::create_MPIReceive(){}

BOOST_PYTHON_MODULE(nengo_mpi)
{
    bpy::class_<PythonMpiSimulatorChunk>("PythonMpiSimulatorChunk")
        .def("add_signal", &PythonMpiSimulatorChunk::add_signal)
        .def("create_Reset", &PythonMpiSimulatorChunk::create_Reset)
        .def("create_Copy", &PythonMpiSimulatorChunk::create_Copy)
        .def("create_DotInc", &PythonMpiSimulatorChunk::create_DotInc)
        .def("create_ProdUpdate", &PythonMpiSimulatorChunk::create_ProdUpdate)
        .def("create_SimLIF", &PythonMpiSimulatorChunk::create_SimLIF)
        .def("create_SimLIFRate", &PythonMpiSimulatorChunk::create_SimLIFRate)
        .def("create_MPISend", &PythonMpiSimulatorChunk::create_MPISend)
        .def("create_MPIReceive", &PythonMpiSimulatorChunk::create_MPIReceive);

//    class_<Operator>("Operator");
//
//    class_<Reset>("Reset", init<Vector, float>());
//
//    class_<Copy>("Copy", init<Vector, Vector>());
//
//    class_<DotInc>("DotInc", init<Matrix, Vector, Vector>());
//
//    class_<ProdUpdate>("ProdUpdate", init<Matrix, Vector, Vector, Vector>());
//
//    class_<SimLIF>("SimLIF", init<int, float, float, float, Vector, Vector>());
//
//    class_<SimLIFRate>("SimLIFRate", init<int, float, float, float, Vector, Vector>());
//
//    class_<MPISend>("MPISend", init<>());
//
//    class_<MPIReceive>("MPIReceive", init<>());
}

