#include <iostream>
#include <vector>

#include "python.hpp"

Vector* ndarray_to_vector(bpyn::array a){
    int ndim = bpy::extract<int>(a.attr("ndim"));
    int size = 1;
    std::vector<int> shape(ndim);
    bpy::object python_shape = a.attr("shape");
    for(unsigned i = 0; i < ndim; i++){ 
        shape[i] = bpy::extract<int>(python_shape[i]);
        size *= shape[i];
    }

    Vector* ret = new Vector(size);
    for(unsigned i = 0; i < size; i++){
        (*ret)(i) = bpy::extract<float>(a[i]);
    }

#ifdef _DEBUG
    std::cout << "Ndim:" << ndim << std::endl;
    std::cout << "Shape:" << std::endl;
    std::cout << "(";
    for(unsigned i = 0; i < ndim; i++){
        std::cout << (*ret)(i);
    }
    std::cout << ")" << std::endl;
    std::cout << "Size:" << size << std::endl;
    std::cout << "Value:" << std::endl;
    for(unsigned i = 0; i < size; i++){
        std::cout << i << ": " << (*ret)(i) << std::endl;
    }
#endif

    return ret;
}

//Matrix* ndarray_to_matrix(bpyn::array a){}

void PythonMpiSimulatorChunk::add_signal(bpy::object key, bpyn::array sig){
    // This is how to get floats
    //std::cout << "First array item" << bpy::extract<float>(sig[0]) << std::endl;
    //
    Vector* vec = ndarray_to_vector(sig);
    //mpi_sim_chunk.add_signal(bpy::extract<key_type>(key), vec);

    // And this is how to get ints
    std::cout << "First array item" << bpy::extract<int>(sig[0].attr("__int__")()) << std::endl;
}

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
    bpy::numeric::array::set_module_and_type("numpy", "ndarray");
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

