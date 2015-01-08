#include "python.hpp"

bool hasattr(bpy::object obj, string const &attrName) {
      return PyObject_HasAttrString(obj.ptr(), attrName.c_str());
 }

Matrix* ndarray_to_matrix(bpyn::array a){

    int ndim = bpy::extract<int>(a.attr("ndim"));

    if (ndim == 1){

        int size = bpy::extract<int>(a.attr("size"));
        Matrix* ret = new Matrix(size, 1);
        for(unsigned i = 0; i < size; i++){
            (*ret)(i, 0) = bpy::extract<floattype>(a[i]);
        }

        return ret;

    }else{
        int size = bpy::extract<int>(a.attr("size"));
        vector<int> shape(ndim);
        vector<int> strides(ndim);
        bpy::object python_shape = a.attr("shape");
        bpy::object python_strides = a.attr("strides");

        for(unsigned i = 0; i < ndim; i++){
            shape[i] = bpy::extract<int>(python_shape[i]);
            strides[i] = bpy::extract<int>(python_strides[i]);
        }

        for(unsigned i = 0; i < ndim; i++){
            dbg("shape: " << shape[i]);
            dbg("strides: " << strides[i]);
        }

        Matrix* ret = new Matrix(shape[0], shape[1]);
        for(unsigned i = 0; i < shape[0]; i++){
            for(unsigned j = 0; j < shape[1]; j++){
                (*ret)(i, j) = bpy::extract<floattype>(a[i][j]);
            }
        }

        return ret;
    }
}

Matrix* list_to_matrix(bpy::list l){

    dbg("Extracting matrix from python list:");

    int length = bpy::len(l);
    Matrix* ret = new Matrix(length, 1);
    for(unsigned i = 0; i < length; i++){
        (*ret)(i, 0) = bpy::extract<floattype>(l[i]);
    }

    dbg("Length:" << length);
    dbg("Value:");
    dbg(*ret << endl);

    return ret;
}

PythonMpiSimulator::PythonMpiSimulator(){
    master_chunk = NULL;
}

PythonMpiSimulator::PythonMpiSimulator(bpy::object num_components, bpy::object dt){

    int c_num_components = bpy::extract<int>(num_components);
    float c_dt = bpy::extract<float>(dt);

    mpi_sim = MpiSimulator(c_num_components, c_dt);
    master_chunk = mpi_sim.master_chunk;
}

void PythonMpiSimulator::finalize(){
    mpi_sim.finalize();
}

void PythonMpiSimulator::run_n_steps(bpy::object pysteps){
    int steps = bpy::extract<int>(pysteps);
    mpi_sim.run_n_steps(steps);
}

bpy::object PythonMpiSimulator::get_probe_data(bpy::object probe_key, bpy::object make_array){
    key_type c_probe_key;
    c_probe_key = bpy::extract<key_type>(probe_key);
    vector<Matrix*> data = mpi_sim.get_probe_data(c_probe_key);

    bpy::list py_list;
    vector<Matrix*>::const_iterator it;
    for(it = data.begin(); it != data.end(); ++it){

        bpy::object a;
        if((*it)->size1() == 1){
            a = make_array((*it)->size2());
            for(unsigned i=0; i < (*it)->size2(); ++i){
                a[i] = (**it)(0, i);
            }
        }else if((*it)->size2() == 1){
            a = make_array((*it)->size1());
            for(unsigned i=0; i < (*it)->size1(); ++i){
                a[i] = (**it)(i, 0);
            }
        }else{
            a = make_array(bpy::make_tuple((*it)->size1(), (*it)->size2()));
            for(unsigned i=0; i < (*it)->size1(); ++i){
                for(unsigned j=0; j < (*it)->size2(); ++j){
                    // TODO: make sure this goes in the right direction wrt to storage format
                    // (col major vs row major)
                    a[i][j] = (**it)(i, j);
                }
            }
        }


        py_list.append(a);
    }

    return py_list;
}

void PythonMpiSimulator::reset(){
    mpi_sim.reset();
}

void PythonMpiSimulator::write_to_file(string filename){
    mpi_sim.write_to_file(filename);
}

void PythonMpiSimulator::read_from_file(string filename){
    mpi_sim.read_from_file(filename);
}

void PythonMpiSimulator::add_signal(
        bpy::object component, bpy::object key, bpy::object label, bpyn::array data){

    int c_component = bpy::extract<int>(component);
    key_type c_key = bpy::extract<key_type>(key);
    string c_label = bpy::extract<string>(label);
    Matrix* c_data = ndarray_to_matrix(data);

    dbg("Adding signal in C++ simulator");
    dbg("Component: "<< c_component);
    dbg("Key: " << c_key);
    dbg("Label: "<< c_label);
    dbg("Size: (" << c_data->size1() << ", " << c_data->size2() << ")");
    dbg("Value:");
    dbg(*c_data << endl);

    mpi_sim.add_signal(c_component, c_key, c_label, c_data);
}

void PythonMpiSimulator::add_op(bpy::object component, bpy::object op_string){
    int c_component = bpy::extract<int>(component);
    string c_op_string = bpy::extract<string>(op_string);

    mpi_sim.add_op(c_component, c_op_string);
}

void PythonMpiSimulator::add_probe(
        bpy::object component, bpy::object probe_key, bpy::object signal_key, bpy::object period){

    key_type c_component = bpy::extract<key_type>(component);
    key_type c_probe_key = bpy::extract<key_type>(probe_key);
    key_type c_signal_key = bpy::extract<key_type>(signal_key);
    floattype c_period = bpy::extract<floattype>(period);

    mpi_sim.add_probe(c_component, c_probe_key, c_signal_key, c_period);
}

void PythonMpiSimulator::create_PyFunc(bpy::object py_fn, bpy::object t_in){

    bool c_t_in = bpy::extract<bool>(t_in);
    double* time_pointer = c_t_in ? master_chunk->get_time_pointer() : NULL;

    Operator* sim_py_func = new PyFunc(NULL, py_fn, time_pointer);

    master_chunk->add_op(sim_py_func);
}

void PythonMpiSimulator::create_PyFuncO(bpy::object output, bpy::object py_fn, bpy::object t_in){

    key_type output_key = bpy::extract<key_type>(output);
    Matrix* output_mat = master_chunk->get_signal(output_key);

    bool c_t_in = bpy::extract<bool>(t_in);
    double* time_pointer = c_t_in ? master_chunk->get_time_pointer() : NULL;

    Operator* sim_py_func = new PyFunc(output_mat, py_fn, time_pointer);

    master_chunk->add_op(sim_py_func);
}

void PythonMpiSimulator::create_PyFuncI(
        bpy::object py_fn, bpy::object t_in, bpy::object input, bpyn::array py_input){

    key_type input_key = bpy::extract<key_type>(input);
    Matrix* input_mat = master_chunk->get_signal(input_key);

    bool c_t_in = bpy::extract<bool>(t_in);
    double* time_pointer = c_t_in ? master_chunk->get_time_pointer() : NULL;

    Operator* sim_py_func = new PyFunc(NULL, py_fn, time_pointer, input_mat, py_input);

    master_chunk->add_op(sim_py_func);
}

void PythonMpiSimulator::create_PyFuncIO(
        bpy::object output, bpy::object py_fn, bpy::object t_in,
        bpy::object input, bpyn::array py_input){

    key_type output_key = bpy::extract<key_type>(output);
    Matrix* output_mat = master_chunk->get_signal(output_key);

    key_type input_key = bpy::extract<key_type>(input);
    Matrix* input_mat = master_chunk->get_signal(input_key);

    bool c_t_in = bpy::extract<bool>(t_in);
    double* time_pointer = c_t_in ? master_chunk->get_time_pointer() : NULL;

    Operator* sim_py_func = new PyFunc(
        output_mat, py_fn, time_pointer, input_mat, py_input);

    master_chunk->add_op(sim_py_func);
}

string PythonMpiSimulator::to_string() const{
    return mpi_sim.to_string();
}

PyFunc::PyFunc(Matrix* output, bpy::object py_fn, double* time)
    :output(output), py_fn(py_fn), time(time), supply_time(time!=NULL), supply_input(false), input(NULL), py_input(0.0){
}

PyFunc::PyFunc(Matrix* output, bpy::object py_fn, double* time, Matrix* input, bpyn::array py_input)
    :output(output), py_fn(py_fn),  time(time), supply_time(time!=NULL),
     supply_input(true), input(input), py_input(py_input){
}

void PyFunc::operator() (){

    bpy::object py_output;
    if(supply_input){

        // TODO: currently assuming pyfunc only operate on vectors.
        for(unsigned i = 0; i < input->size1(); ++i){
            py_input[i] = (*input)(i, 0);
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
            // TODO: currently assuming pyfunc only operate on vectors.
            for(unsigned i = 0; i < output->size1(); ++i){
                (*output)(i, 0) = bpy::extract<floattype>(py_output[i]);
            }
        }else{
            (*output)(0, 0) = bpy::extract<floattype>(py_output);
        }
    }

    run_dbg(*this);
}

string PyFunc::to_string() const{
    stringstream out;

    out << "PyFunc: " << endl;
    out << "Output: " << endl;
    out << *output << endl << endl;

    return out.str();
}

BOOST_PYTHON_MODULE(mpi_sim)
{
    bpy::numeric::array::set_module_and_type("numpy", "ndarray");
    bpy::class_<PythonMpiSimulator>("PythonMpiSimulator", bpy::init<bpy::object, bpy::object>())
        .def("finalize", &PythonMpiSimulator::finalize)
        .def("run_n_steps", &PythonMpiSimulator::run_n_steps)
        .def("get_probe_data", &PythonMpiSimulator::get_probe_data)
        .def("reset", &PythonMpiSimulator::reset)
        .def("write_to_file", &PythonMpiSimulator::write_to_file)
        .def("read_from_file", &PythonMpiSimulator::read_from_file)
        .def("add_signal", &PythonMpiSimulator::add_signal)
        .def("add_op", &PythonMpiSimulator::add_op)
        .def("add_probe", &PythonMpiSimulator::add_probe)
        .def("create_PyFunc", &PythonMpiSimulator::create_PyFunc)
        .def("create_PyFuncO", &PythonMpiSimulator::create_PyFuncO)
        .def("create_PyFuncI", &PythonMpiSimulator::create_PyFuncI)
        .def("create_PyFuncIO", &PythonMpiSimulator::create_PyFuncIO)
        .def("to_string", &PythonMpiSimulator::to_string);
}

