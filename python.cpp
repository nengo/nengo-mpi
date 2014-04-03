#include <iostream>
#include <vector>

#include "python.hpp"

bool is_vector(bpyn::array a){
    int ndim = bpy::extract<int>(a.attr("ndim"));
    return ndim == 1;
}

Vector* ndarray_to_vector(bpyn::array a){

#ifdef _DEBUG
    std::cout << "Extracting vector:" << std::endl;
#endif

    int size = bpy::extract<int>(a.attr("size"));
    Vector* ret = new Vector(size);
    for(unsigned i = 0; i < size; i++){
        (*ret)(i) = bpy::extract<float>(a[i]);
    }

#ifdef _DEBUG
    std::cout << "Size:" << size << std::endl;
    std::cout << "Value:" << std::endl;
    std::cout << *ret << std::endl << std::endl;
#endif

    return ret;
}

Matrix* ndarray_to_matrix(bpyn::array a){

#ifdef _DEBUG
    std::cout << "Extracting matrix:" << std::endl;
#endif

    int ndim = bpy::extract<int>(a.attr("ndim"));
    int size = bpy::extract<int>(a.attr("size"));
    std::vector<int> shape(ndim);
    std::vector<int> strides(ndim);
    bpy::object python_shape = a.attr("shape");
    bpy::object python_strides = a.attr("strides");
    for(unsigned i = 0; i < ndim; i++){ 
        shape[i] = bpy::extract<int>(python_shape[i]);
        strides[i] = bpy::extract<int>(python_strides[i]);
    }

    Matrix* ret = new Matrix(shape[0], shape[1]);
    for(unsigned i = 0; i < shape[0]; i++){
        for(unsigned j = 0; j < shape[1]; j++){
            (*ret)(i, j) = bpy::extract<float>(a[i][j]);
        }
    }

#ifdef _DEBUG
    std::cout << "Ndim:" << ndim << std::endl;
    std::cout << "Size:" << size << std::endl;
    std::cout << "Shape:" << std::endl;
    std::cout << "(";
    for(unsigned i = 0; i < ndim; i++){
        std::cout << shape[i] << ",";
    }
    std::cout << ")" << std::endl;
    std::cout << "Value:" << std::endl;
    std::cout << *ret << std::endl << std::endl;
#endif

    return ret;
}

void PythonMpiSimulatorChunk::run_n_steps(bpy::object pysteps){
    int steps = bpy::extract<int>(pysteps);
    mpi_sim_chunk.run_n_steps(steps);
}

void PythonMpiSimulatorChunk::add_signal(bpy::object key, bpyn::array sig){
    if( is_vector(sig) ){
        Vector* vec = ndarray_to_vector(sig);
        mpi_sim_chunk.add_vector_signal(bpy::extract<key_type>(key), vec);
    }else{
        Matrix* mat = ndarray_to_matrix(sig);
        mpi_sim_chunk.add_matrix_signal(bpy::extract<key_type>(key), mat);
    }
}

void PythonMpiSimulatorChunk::create_Reset(bpy::object dst, bpy::object value){
    key_type dst_key = bpy::extract<key_type>(dst);
    float c_value = bpy::extract<float>(value);

    Vector* dst_vec = mpi_sim_chunk.get_vector_signal(dst_key);

    Operator* reset = new Reset(dst_vec, c_value);
    mpi_sim_chunk.add_operator(reset);
}

void PythonMpiSimulatorChunk::create_Copy(bpy::object dst, bpy::object src){
    key_type dst_key = bpy::extract<key_type>(dst);
    key_type src_key = bpy::extract<key_type>(src);

    Vector* dst_vec = mpi_sim_chunk.get_vector_signal(dst_key);
    Vector* src_vec = mpi_sim_chunk.get_vector_signal(src_key);

    Operator* copy = new Copy(dst_vec, src_vec);
    mpi_sim_chunk.add_operator(copy);
}

void PythonMpiSimulatorChunk::create_DotInc(bpy::object A, bpy::object X, bpy::object Y){
    key_type A_key = bpy::extract<key_type>(A);
    key_type X_key = bpy::extract<key_type>(X);
    key_type Y_key = bpy::extract<key_type>(Y);

    Matrix* A_mat = mpi_sim_chunk.get_matrix_signal(A_key);
    Vector* X_vec = mpi_sim_chunk.get_vector_signal(X_key);
    Vector* Y_vec = mpi_sim_chunk.get_vector_signal(Y_key);

    Operator* dot_inc = new DotInc(A_mat, X_vec, Y_vec);
    mpi_sim_chunk.add_operator(dot_inc);
}

void PythonMpiSimulatorChunk::create_ProdUpdate(bpy::object A, bpy::object X, bpy::object B, bpy::object Y){
    key_type A_key = bpy::extract<key_type>(A);
    key_type X_key = bpy::extract<key_type>(X);
    key_type B_key = bpy::extract<key_type>(B);
    key_type Y_key = bpy::extract<key_type>(Y);

    Matrix* A_mat = mpi_sim_chunk.get_matrix_signal(A_key);
    Vector* X_vec = mpi_sim_chunk.get_vector_signal(X_key);
    Vector* B_vec = mpi_sim_chunk.get_vector_signal(B_key);
    Vector* Y_vec = mpi_sim_chunk.get_vector_signal(Y_key);

    Operator* prod_update = new ProdUpdate(A_mat, X_vec, B_vec, Y_vec);
    mpi_sim_chunk.add_operator(prod_update);

}

void PythonMpiSimulatorChunk::create_SimLIF(bpy::object n_neurons, bpy::object tau_rc, 
    bpy::object tau_ref, bpy::object dt, bpy::object J, bpy::object output){

    int c_n_neurons = bpy::extract<int>(n_neurons);
    float c_tau_rc = bpy::extract<float>(tau_rc);
    float c_tau_ref = bpy::extract<float>(tau_ref);
    float c_dt = bpy::extract<float>(dt);

    key_type J_key = bpy::extract<key_type>(J);
    key_type output_key = bpy::extract<key_type>(output);

    Vector* J_vec = mpi_sim_chunk.get_vector_signal(J_key);
    Vector* output_vec = mpi_sim_chunk.get_vector_signal(output_key);

    Operator* sim_lif = new SimLIF(c_n_neurons, c_tau_rc, c_tau_ref, c_dt, J_vec, output_vec);
    mpi_sim_chunk.add_operator(sim_lif);
}

void PythonMpiSimulatorChunk::create_SimLIFRate(bpy::object n_neurons, bpy::object tau_rc, 
    bpy::object tau_ref, bpy::object dt, bpy::object J, bpy::object output){

    int c_n_neurons = bpy::extract<int>(n_neurons);
    float c_tau_rc = bpy::extract<float>(tau_rc);
    float c_tau_ref = bpy::extract<float>(tau_ref);
    float c_dt = bpy::extract<float>(dt);

    key_type J_key = bpy::extract<key_type>(J);
    key_type output_key = bpy::extract<key_type>(output);

    Vector* J_vec = mpi_sim_chunk.get_vector_signal(J_key);
    Vector* output_vec = mpi_sim_chunk.get_vector_signal(output_key);

    Operator* sim_lif_rate = new SimLIFRate(c_n_neurons, c_tau_rc, c_tau_ref, c_dt, J_vec, output_vec);
    mpi_sim_chunk.add_operator(sim_lif_rate);
}

void PythonMpiSimulatorChunk::create_MPISend(){}

void PythonMpiSimulatorChunk::create_MPIReceive(){}

void PythonMpiSimulatorChunk::create_PyFunc(bpy::object output, bpy::object py_fn, 
    bpy::object t_in, bpy::object input){

    bool c_t_in = bpy::extract<bool>(t_in);

    key_type output_key = bpy::extract<key_type>(output);
    key_type input_key = bpy::extract<key_type>(output);

    bool use_input = input_key != -1;

    Vector* output_vec = mpi_sim_chunk.get_vector_signal(output_key);

    Vector* input_vec;
    if (use_input){
        input_vec = mpi_sim_chunk.get_vector_signal(input_key);
    }

    Operator* sim_py_func = new PyFunc(output_vec, py_fn, c_t_in, input_vec);
    mpi_sim_chunk.add_operator(sim_py_func);
}

//PyFunc::PyFunc(Vector* output, bpy::object py_fn, bpy::object t_in)
//    :output(output), py_fn(py_fn), supply_time(t_in), supply_input(false), input(NULL){
//}

PyFunc::PyFunc(Vector* output, bpy::object py_fn, bool t_in, Vector* input)
    :output(output), py_fn(py_fn), supply_time(t_in), supply_input(input!=NULL), input(input){
}

void PyFunc::operator() (){
    //If supplying time, convert time signal to python objectr
    //If supplying input, convert input signal to python object
    //Call py_fn
    //Store result in output
}


BOOST_PYTHON_MODULE(nengo_mpi)
{
    bpy::numeric::array::set_module_and_type("numpy", "ndarray");
    bpy::class_<PythonMpiSimulatorChunk>("PythonMpiSimulatorChunk")
        .def("run_n_steps", &PythonMpiSimulatorChunk::run_n_steps)
        .def("add_signal", &PythonMpiSimulatorChunk::add_signal)
        .def("create_Reset", &PythonMpiSimulatorChunk::create_Reset)
        .def("create_Copy", &PythonMpiSimulatorChunk::create_Copy)
        .def("create_DotInc", &PythonMpiSimulatorChunk::create_DotInc)
        .def("create_ProdUpdate", &PythonMpiSimulatorChunk::create_ProdUpdate)
        .def("create_SimLIF", &PythonMpiSimulatorChunk::create_SimLIF)
        .def("create_SimLIFRate", &PythonMpiSimulatorChunk::create_SimLIFRate)
        .def("create_MPISend", &PythonMpiSimulatorChunk::create_MPISend)
        .def("create_MPIReceive", &PythonMpiSimulatorChunk::create_MPIReceive)
        .def("create_PyFunc", &PythonMpiSimulatorChunk::create_PyFunc);
}

