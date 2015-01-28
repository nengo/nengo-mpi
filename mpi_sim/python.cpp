#include "python.hpp"

bool hasattr(bpy::object obj, string const &attrName) {
      return PyObject_HasAttrString(obj.ptr(), attrName.c_str());
 }

BaseMatrix* ndarray_to_matrix(bpyn::array a){

    int ndim = bpy::extract<int>(a.attr("ndim"));

    if (ndim == 1){

        int size = bpy::extract<int>(a.attr("size"));
        BaseMatrix* ret = new BaseMatrix(size, 1);
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

        BaseMatrix* ret = new BaseMatrix(shape[0], shape[1]);
        for(unsigned i = 0; i < shape[0]; i++){
            for(unsigned j = 0; j < shape[1]; j++){
                (*ret)(i, j) = bpy::extract<floattype>(a[i][j]);
            }
        }

        return ret;
    }
}

BaseMatrix* list_to_matrix(bpy::list l){

    dbg("Extracting matrix from python list:");

    int length = bpy::len(l);
    BaseMatrix* ret = new BaseMatrix(length, 1);
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

PythonMpiSimulator::PythonMpiSimulator(
        bpy::object num_components, bpy::object dt, bpy::object out_filename):
    mpi_sim(bpy::extract<int>(num_components), bpy::extract<float>(dt), bpy::extract<string>(out_filename)){

    master_chunk = mpi_sim.master_chunk;
}

void PythonMpiSimulator::finalize(){
    mpi_sim.finalize();
}

void PythonMpiSimulator::run_n_steps(bpy::object pysteps, bpy::object progress){
    int c_steps = bpy::extract<int>(pysteps);
    bool c_progress = bpy::extract<bool>(progress);

    mpi_sim.run_n_steps(c_steps, c_progress);
}

bpy::object PythonMpiSimulator::get_probe_data(bpy::object probe_key, bpy::object make_array){
    key_type c_probe_key = bpy::extract<key_type>(probe_key);
    vector<BaseMatrix*> data = mpi_sim.get_probe_data(c_probe_key);

    bpy::list py_list;
    vector<BaseMatrix*>::const_iterator it;
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

void PythonMpiSimulator::add_signal(
        bpy::object component, bpy::object key, bpy::object label, bpyn::array data){

    int c_component = bpy::extract<int>(component);
    key_type c_key = bpy::extract<key_type>(key);
    string c_label = bpy::extract<string>(label);
    BaseMatrix* c_data = ndarray_to_matrix(data);

    dbg("Adding base signal in C++ simulator");
    dbg("Component: "<< c_component);
    dbg("Key: " << c_key);
    dbg("Label: "<< c_label);
    dbg("Size: (" << c_data->size1() << ", " << c_data->size2() << ")");
    dbg("Value:");
    dbg(*c_data << endl << endl);

    mpi_sim.add_base_signal(c_component, c_key, c_label, c_data);
}

void PythonMpiSimulator::add_op(bpy::object component, bpy::object op_string){
    int c_component = bpy::extract<int>(component);
    string c_op_string = bpy::extract<string>(op_string);

    mpi_sim.add_op(c_component, c_op_string);
}

void PythonMpiSimulator::add_probe(
        bpy::object component, bpy::object probe_key, bpy::object signal_string, bpy::object period){

    int c_component = bpy::extract<int>(component);
    key_type c_probe_key = bpy::extract<key_type>(probe_key);
    string c_signal_string = bpy::extract<string>(signal_string);
    floattype c_period = bpy::extract<floattype>(period);

    mpi_sim.add_probe(c_component, c_probe_key, c_signal_string, c_period);
}

void PythonMpiSimulator::create_PyFunc(bpy::object py_fn, bpy::object t_in){

    bool c_t_in = bpy::extract<bool>(t_in);
    double* time_pointer = c_t_in ? master_chunk->get_time_pointer() : NULL;

    Operator* sim_py_func = new PyFunc(py_fn, time_pointer);

    master_chunk->add_op(sim_py_func);
}

void PythonMpiSimulator::create_PyFuncI(
        bpy::object py_fn, bpy::object t_in, bpy::object input, bpyn::array py_input){

    string input_signal = bpy::extract<string>(input);
    Matrix input_mat = master_chunk->get_signal(input_signal);

    bool c_t_in = bpy::extract<bool>(t_in);
    double* time_pointer = c_t_in ? master_chunk->get_time_pointer() : NULL;

    Operator* sim_py_func = new PyFunc(py_fn, time_pointer, input_mat, py_input);

    master_chunk->add_op(sim_py_func);
}

void PythonMpiSimulator::create_PyFuncO(bpy::object py_fn, bpy::object t_in, bpy::object output){

    string output_signal = bpy::extract<string>(output);
    Matrix output_mat = master_chunk->get_signal(output_signal);

    bool c_t_in = bpy::extract<bool>(t_in);
    double* time_pointer = c_t_in ? master_chunk->get_time_pointer() : NULL;

    Operator* sim_py_func = new PyFunc(py_fn, time_pointer, output_mat);

    master_chunk->add_op(sim_py_func);
}


void PythonMpiSimulator::create_PyFuncIO(bpy::object py_fn, bpy::object t_in,
                                         bpy::object input, bpyn::array py_input,
                                         bpy::object output){

    string output_signal = bpy::extract<string>(output);
    Matrix output_mat = master_chunk->get_signal(output_signal);

    string input_signal = bpy::extract<string>(input);
    Matrix input_mat = master_chunk->get_signal(input_signal);

    bool c_t_in = bpy::extract<bool>(t_in);
    double* time_pointer = c_t_in ? master_chunk->get_time_pointer() : NULL;

    Operator* sim_py_func = new PyFunc(
        py_fn, time_pointer, input_mat, py_input, output_mat);

    master_chunk->add_op(sim_py_func);
}

string PythonMpiSimulator::to_string() const{
    return mpi_sim.to_string();
}

PyFunc::PyFunc(bpy::object py_fn, double* time)
    :py_fn(py_fn), time(time), supply_time(time!=NULL),
     supply_input(false), get_output(false), input(null_matrix, null_slice, null_slice),
     py_input(0.0), output(null_matrix, null_slice, null_slice){
}

PyFunc::PyFunc(bpy::object py_fn, double* time, Matrix output)
    :py_fn(py_fn), time(time), supply_time(time!=NULL),
     supply_input(false), get_output(true), input(null_matrix, null_slice, null_slice),
     py_input(0.0), output(output){
}

PyFunc::PyFunc(bpy::object py_fn, double* time, Matrix input, bpyn::array py_input)
    :py_fn(py_fn), time(time), supply_time(time!=NULL),
     supply_input(true), get_output(false), input(input),
     py_input(py_input), output(null_matrix, null_slice, null_slice){
}

PyFunc::PyFunc(bpy::object py_fn, double* time, Matrix input, bpyn::array py_input, Matrix output)
    :py_fn(py_fn), time(time), supply_time(time!=NULL),
     supply_input(true), get_output(true), input(input),
     py_input(py_input), output(output){
}

void PyFunc::operator() (){

    bpy::object py_output;
    if(supply_input){

        // TODO: currently assuming pyfunc only operate on vectors.
        for(unsigned i = 0; i < input.size1(); ++i){
            py_input[i] = input(i, 0);
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

    if(get_output){
        if(hasattr(py_output, "ndim")){
            // TODO: currently assuming pyfunc only operate on vectors.
            for(unsigned i = 0; i < output.size1(); ++i){
                output(i, 0) = bpy::extract<floattype>(py_output[i]);
            }
        }else{
            output(0, 0) = bpy::extract<floattype>(py_output);
        }
    }

    run_dbg(*this);
}

string PyFunc::to_string() const{
    stringstream out;

    out << "PyFunc: " << endl;
    out << "Output: " << endl;
    out << output << endl << endl;

    return out.str();
}

BOOST_PYTHON_MODULE(mpi_sim)
{
    bpy::numeric::array::set_module_and_type("numpy", "ndarray");
    bpy::class_<PythonMpiSimulator>("PythonMpiSimulator", bpy::init<bpy::object, bpy::object, bpy::object>())
        .def("finalize", &PythonMpiSimulator::finalize)
        .def("run_n_steps", &PythonMpiSimulator::run_n_steps)
        .def("get_probe_data", &PythonMpiSimulator::get_probe_data)
        .def("reset", &PythonMpiSimulator::reset)
        .def("add_signal", &PythonMpiSimulator::add_signal)
        .def("add_op", &PythonMpiSimulator::add_op)
        .def("add_probe", &PythonMpiSimulator::add_probe)
        .def("create_PyFunc", &PythonMpiSimulator::create_PyFunc)
        .def("create_PyFuncO", &PythonMpiSimulator::create_PyFuncO)
        .def("create_PyFuncI", &PythonMpiSimulator::create_PyFuncI)
        .def("create_PyFuncIO", &PythonMpiSimulator::create_PyFuncIO)
        .def("to_string", &PythonMpiSimulator::to_string);
}

