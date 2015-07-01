#include "mpi_python.hpp"

NativeMpiSimulator::NativeMpiSimulator(){}

NativeMpiSimulator::NativeMpiSimulator(bpy::object num_components, bpy::object dt){

    int c_num_components = bpy::extract<int>(num_components);
    dtype c_dt = bpy::extract<dtype>(dt);
    sim = unique_ptr<Simulator>(new MpiSimulator(c_num_components, c_dt, false));
}

BOOST_PYTHON_MODULE(mpi_sim)
{
    bpy::numeric::array::set_module_and_type("numpy", "ndarray");
    bpy::class_<NativeMpiSimulator, boost::noncopyable>(
            "NativeMpiSimulator", bpy::init<bpy::object, bpy::object>())
        .def("load_network", &NativeMpiSimulator::load_network)
        .def("finalize_build", &NativeMpiSimulator::finalize_build)
        .def("run_n_steps", &NativeMpiSimulator::run_n_steps)
        .def("get_probe_data", &NativeMpiSimulator::get_probe_data)
        .def("reset", &NativeMpiSimulator::reset)
        .def("create_PyFunc", &NativeMpiSimulator::create_PyFunc)
        .def("create_PyFuncO", &NativeMpiSimulator::create_PyFuncO)
        .def("create_PyFuncI", &NativeMpiSimulator::create_PyFuncI)
        .def("create_PyFuncIO", &NativeMpiSimulator::create_PyFuncIO)
        .def("to_string", &NativeMpiSimulator::to_string);
}

