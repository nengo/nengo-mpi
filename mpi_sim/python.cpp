#include <iostream>
#include <vector>
#include <list>

#include "python.hpp"

bool is_vector(bpyn::array a){
    int ndim = bpy::extract<int>(a.attr("ndim"));
    return ndim == 1;
}

Vector* ndarray_to_vector(bpyn::array a){

#ifdef _DEBUG
    std::cout << "Extracting vector from ndarray:" << std::endl;
#endif

    int size = bpy::extract<int>(a.attr("size"));
    Vector* ret = new Vector(size);
    for(unsigned i = 0; i < size; i++){
        (*ret)(i) = bpy::extract<floattype>(a[i]);
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
    std::cout << "Extracting matrix from ndarray:" << std::endl;
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
            (*ret)(i, j) = bpy::extract<floattype>(a[i][j]);
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

Vector* list_to_vector(bpy::list l){

#ifdef _DEBUG
    std::cout << "Extracting from python list vector:" << std::endl;
#endif

    int length = bpy::len(l);
    Vector* ret = new Vector(length);
    for(unsigned i = 0; i < length; i++){
        (*ret)(i) = bpy::extract<floattype>(l[i]);
    }

#ifdef _DEBUG
    std::cout << "Length:" << length << std::endl;
    std::cout << "Value:" << std::endl;
    std::cout << *ret << std::endl << std::endl;
#endif

    return ret;
}

bool hasattr(bpy::object obj, string const &attrName) {
      return PyObject_HasAttrString(obj.ptr(), attrName.c_str());
 }

PythonMpiSimulatorChunk::PythonMpiSimulatorChunk(){
}

PythonMpiSimulatorChunk::PythonMpiSimulatorChunk(double dt)
    :mpi_sim_chunk(dt){
}

void PythonMpiSimulatorChunk::run_n_steps(bpy::object pysteps){
    int steps = bpy::extract<int>(pysteps);
    mpi_sim_chunk.run_n_steps(steps);
}

void PythonMpiSimulatorChunk::add_vector_signal(bpy::object key, bpyn::array sig){
    Vector* vec = ndarray_to_vector(sig);
    mpi_sim_chunk.add_vector_signal(bpy::extract<key_type>(key), vec);
}

void PythonMpiSimulatorChunk::add_matrix_signal(bpy::object key, bpyn::array sig){
    Matrix* mat = ndarray_to_matrix(sig);
    mpi_sim_chunk.add_matrix_signal(bpy::extract<key_type>(key), mat);
}

bpy::object PythonMpiSimulatorChunk::get_probe_data(bpy::object probe_key, bpy::object make_array){
    key_type c_probe_key ;
    c_probe_key = bpy::extract<key_type>(probe_key);
    Probe<Vector>* probe = mpi_sim_chunk.get_probe(c_probe_key);
    list<Vector*> data = probe->get_data();

    bpy::list py_list;
    list<Vector*>::const_iterator it;
    for(it = data.begin(); it != data.end(); ++it){

        bpy::object a = make_array((*it)->size());
        for(unsigned i=0; i < (*it)->size(); ++i){
            a[i] = (**it)[i];
        }

        py_list.append(a);
    }

    return py_list;
}

void PythonMpiSimulatorChunk::create_Probe(bpy::object key, bpy::object signal, bpy::object period){
    key_type signal_key = bpy::extract<key_type>(signal);
    Vector* signal_vec = mpi_sim_chunk.get_vector_signal(signal_key);
    int c_period = bpy::extract<int>(period);

    Probe<Vector>* probe = new Probe<Vector>(signal_vec, c_period);

    key_type c_key = bpy::extract<key_type>(key);
    mpi_sim_chunk.add_probe(c_key, probe);
}

void PythonMpiSimulatorChunk::create_Reset(bpy::object dst, bpy::object value){
    key_type dst_key = bpy::extract<key_type>(dst);
    floattype c_value = bpy::extract<floattype>(value);

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

void PythonMpiSimulatorChunk::create_DotIncMV(bpy::object A, bpy::object X, bpy::object Y){
    key_type A_key = bpy::extract<key_type>(A);
    key_type X_key = bpy::extract<key_type>(X);
    key_type Y_key = bpy::extract<key_type>(Y);

    Matrix* A_mat = mpi_sim_chunk.get_matrix_signal(A_key);
    Vector* X_vec = mpi_sim_chunk.get_vector_signal(X_key);
    Vector* Y_vec = mpi_sim_chunk.get_vector_signal(Y_key);

    Operator* dot_inc = new DotIncMV(A_mat, X_vec, Y_vec);
    mpi_sim_chunk.add_operator(dot_inc);
}

void PythonMpiSimulatorChunk::create_DotIncVV(bpy::object A, bpy::object X, bpy::object Y){
    key_type A_key = bpy::extract<key_type>(A);
    key_type X_key = bpy::extract<key_type>(X);
    key_type Y_key = bpy::extract<key_type>(Y);

    Vector* A_vec = mpi_sim_chunk.get_vector_signal(A_key);
    Vector* X_vec = mpi_sim_chunk.get_vector_signal(X_key);
    Vector* Y_vec = mpi_sim_chunk.get_vector_signal(Y_key);

    Operator* dot_inc = new DotIncVV(A_vec, X_vec, Y_vec);
    mpi_sim_chunk.add_operator(dot_inc);
}

void PythonMpiSimulatorChunk::create_ProdUpdate(bpy::object B, bpy::object Y){
    key_type B_key = bpy::extract<key_type>(B);
    key_type Y_key = bpy::extract<key_type>(Y);

    Vector* B_vec = mpi_sim_chunk.get_vector_signal(B_key);
    Vector* Y_vec = mpi_sim_chunk.get_vector_signal(Y_key);

    Operator* prod_update = new ProdUpdate(B_vec, Y_vec);
    mpi_sim_chunk.add_operator(prod_update);
}

void PythonMpiSimulatorChunk::create_Filter(bpy::object input, bpy::object output,
                                            bpy::list numer, bpy::list denom){

    key_type input_key = bpy::extract<key_type>(input);
    key_type output_key = bpy::extract<key_type>(output);

    Vector* input_vec = mpi_sim_chunk.get_vector_signal(input_key);
    Vector* output_vec = mpi_sim_chunk.get_vector_signal(output_key);

    Vector* numer_vec = list_to_vector(numer);
    Vector* denom_vec = list_to_vector(denom);

    Operator* filter = new Filter(input_vec, output_vec, numer_vec, denom_vec);
    mpi_sim_chunk.add_operator(filter);
}

void PythonMpiSimulatorChunk::create_SimLIF(bpy::object n_neurons, bpy::object tau_rc,
    bpy::object tau_ref, bpy::object dt, bpy::object J, bpy::object output){

    int c_n_neurons = bpy::extract<int>(n_neurons);
    floattype c_tau_rc = bpy::extract<floattype>(tau_rc);
    floattype c_tau_ref = bpy::extract<floattype>(tau_ref);
    floattype c_dt = bpy::extract<floattype>(dt);

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
    floattype c_tau_rc = bpy::extract<floattype>(tau_rc);
    floattype c_tau_ref = bpy::extract<floattype>(tau_ref);
    floattype c_dt = bpy::extract<floattype>(dt);

    key_type J_key = bpy::extract<key_type>(J);
    key_type output_key = bpy::extract<key_type>(output);

    Vector* J_vec = mpi_sim_chunk.get_vector_signal(J_key);
    Vector* output_vec = mpi_sim_chunk.get_vector_signal(output_key);

    Operator* sim_lif_rate = new SimLIFRate(c_n_neurons, c_tau_rc, c_tau_ref, c_dt, J_vec, output_vec);
    mpi_sim_chunk.add_operator(sim_lif_rate);
}

void PythonMpiSimulatorChunk::create_MPISend(){}

void PythonMpiSimulatorChunk::create_MPIRecv(){}

void PythonMpiSimulatorChunk::create_PyFunc(bpy::object py_fn, bpy::object t_in){

    bool c_t_in = bpy::extract<bool>(t_in);
    double* time_pointer = c_t_in ? mpi_sim_chunk.get_time_pointer() : NULL;

    Operator* sim_py_func = new PyFunc(NULL, py_fn, time_pointer);

    mpi_sim_chunk.add_operator(sim_py_func);
}

void PythonMpiSimulatorChunk::create_PyFuncO(bpy::object output, bpy::object py_fn, bpy::object t_in){

    key_type output_key = bpy::extract<key_type>(output);
    Vector* output_vec = mpi_sim_chunk.get_vector_signal(output_key);

    bool c_t_in = bpy::extract<bool>(t_in);
    double* time_pointer = c_t_in ? mpi_sim_chunk.get_time_pointer() : NULL;

    Operator* sim_py_func = new PyFunc(output_vec, py_fn, time_pointer);

    mpi_sim_chunk.add_operator(sim_py_func);
}

void PythonMpiSimulatorChunk::create_PyFuncI(
        bpy::object py_fn, bpy::object t_in, bpy::object input, bpyn::array py_input){

    key_type input_key = bpy::extract<key_type>(input);
    Vector* input_vec = mpi_sim_chunk.get_vector_signal(input_key);

    bool c_t_in = bpy::extract<bool>(t_in);
    double* time_pointer = c_t_in ? mpi_sim_chunk.get_time_pointer() : NULL;

    Operator* sim_py_func = new PyFunc(NULL, py_fn, time_pointer, input_vec, py_input);

    mpi_sim_chunk.add_operator(sim_py_func);
}

void PythonMpiSimulatorChunk::create_PyFuncIO(
        bpy::object output, bpy::object py_fn, bpy::object t_in,
        bpy::object input, bpyn::array py_input){

    key_type output_key = bpy::extract<key_type>(output);
    Vector* output_vec = mpi_sim_chunk.get_vector_signal(output_key);

    key_type input_key = bpy::extract<key_type>(input);
    Vector* input_vec = mpi_sim_chunk.get_vector_signal(input_key);

    bool c_t_in = bpy::extract<bool>(t_in);
    double* time_pointer = c_t_in ? mpi_sim_chunk.get_time_pointer() : NULL;

    Operator* sim_py_func = new PyFunc(
        output_vec, py_fn, time_pointer, input_vec, py_input);

    mpi_sim_chunk.add_operator(sim_py_func);
}


PyFunc::PyFunc(Vector* output, bpy::object py_fn, double* time)
    :output(output), py_fn(py_fn), time(time), supply_time(time!=NULL), supply_input(false), input(NULL), py_input(0.0){
}

PyFunc::PyFunc(Vector* output, bpy::object py_fn, double* time, Vector* input, bpyn::array py_input)
    :output(output), py_fn(py_fn),  time(time), supply_time(time!=NULL), supply_input(true),
    input(input), py_input(py_input){
}

void PyFunc::operator() (){

    bpy::object py_output;
    if(supply_input){
        for(unsigned i = 0; i < input->size(); ++i){
            py_input[i] = (*input)[i];
        }

        if(supply_time){
            py_output = py_fn(*time, py_input);
        }else{
            py_output = py_fn(py_input);
        }
    }else{
        if(supply_time){
            py_output = py_fn(*time);
        }else{
            py_output = py_fn();
        }
    }

    if(output){
        if(hasattr(py_output, "ndim")){
            for(unsigned i = 0; i < output->size(); ++i){
                (*output)[i] = bpy::extract<floattype>(py_output[i]);
            }
        }else{
            (*output)[0] = bpy::extract<floattype>(py_output);
        }
    }

#ifdef _RUN_DEBUG
    cout << *this;
#endif
}

void PyFunc::print(ostream &out) const{
    out << "PyFunc: " << endl;
    out << "Output: " << endl;
    out << *output << endl << endl;
}

BOOST_PYTHON_MODULE(mpi_sim)
{
    bpy::numeric::array::set_module_and_type("numpy", "ndarray");
    bpy::class_<PythonMpiSimulatorChunk>("PythonMpiSimulatorChunk", bpy::init<>())
        .def(bpy::init<double>())
        .def("run_n_steps", &PythonMpiSimulatorChunk::run_n_steps)
        .def("add_vector_signal", &PythonMpiSimulatorChunk::add_vector_signal)
        .def("add_matrix_signal", &PythonMpiSimulatorChunk::add_matrix_signal)
        .def("get_probe_data", &PythonMpiSimulatorChunk::get_probe_data)
        .def("create_Probe", &PythonMpiSimulatorChunk::create_Probe)
        .def("create_Reset", &PythonMpiSimulatorChunk::create_Reset)
        .def("create_Copy", &PythonMpiSimulatorChunk::create_Copy)
        .def("create_DotIncMV", &PythonMpiSimulatorChunk::create_DotIncMV)
        .def("create_DotIncVV", &PythonMpiSimulatorChunk::create_DotIncVV)
        .def("create_ProdUpdate", &PythonMpiSimulatorChunk::create_ProdUpdate)
        .def("create_Filter", &PythonMpiSimulatorChunk::create_Filter)
        .def("create_SimLIF", &PythonMpiSimulatorChunk::create_SimLIF)
        .def("create_SimLIFRate", &PythonMpiSimulatorChunk::create_SimLIFRate)
        .def("create_MPISend", &PythonMpiSimulatorChunk::create_MPISend)
        .def("create_MPIRecv", &PythonMpiSimulatorChunk::create_MPIRecv)
        .def("create_PyFunc", &PythonMpiSimulatorChunk::create_PyFunc)
        .def("create_PyFuncO", &PythonMpiSimulatorChunk::create_PyFuncO)
        .def("create_PyFuncI", &PythonMpiSimulatorChunk::create_PyFuncI)
        .def("create_PyFuncIO", &PythonMpiSimulatorChunk::create_PyFuncIO);
}

