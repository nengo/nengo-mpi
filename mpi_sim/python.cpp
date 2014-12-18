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
}

string PythonMpiSimulator::to_string() const{
    return mpi_sim.to_string();
}

PythonMpiSimulatorChunk* PythonMpiSimulator::add_chunk(){
    MpiSimulatorChunk* mpi_sim_chunk = mpi_sim.add_chunk();
    PythonMpiSimulatorChunk* py_chunk = new PythonMpiSimulatorChunk(mpi_sim_chunk);
    py_chunks.push_back(py_chunk);
    return py_chunk;
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
    vector<Matrix*>* data = mpi_sim.get_probe_data(c_probe_key);

    bpy::list py_list;
    vector<Matrix*>::const_iterator it;
    for(it = data->begin(); it != data->end(); ++it){

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

void PythonMpiSimulator::write_to_file(string filename){
    mpi_sim.write_to_file(filename);
}

void PythonMpiSimulator::read_from_file(string filename){
    mpi_sim.read_from_file(filename);
}

PythonMpiSimulatorChunk::PythonMpiSimulatorChunk(){
}

PythonMpiSimulatorChunk::PythonMpiSimulatorChunk(MpiSimulatorChunk* mpi_sim_chunk)
    :mpi_sim_chunk(mpi_sim_chunk){
}

string PythonMpiSimulatorChunk::to_string() const{
    return mpi_sim_chunk->to_string();
}

void PythonMpiSimulatorChunk::add_signal(bpy::object key, bpyn::array sig, bpy::object label){

    Matrix* mat = ndarray_to_matrix(sig);
    key_type c_key = bpy::extract<key_type>(key);
    string c_label = bpy::extract<string>(label);

    dbg("Adding signal in C++ simulator");
    dbg("Label: "<< c_label);
    dbg("Key: " << c_key);
    dbg("Size: (" << mat->size1() << ", " << mat->size2() << ")");
    dbg("Value:");
    dbg(*mat << endl);

    mpi_sim_chunk->add_signal(c_key, c_label, mat);
}

void PythonMpiSimulatorChunk::create_Probe(bpy::object key, bpy::object signal, bpy::object period){
    key_type signal_key = bpy::extract<key_type>(signal);
    Matrix* signal_mat = mpi_sim_chunk->get_signal(signal_key);
    float c_period = bpy::extract<float>(period);

    Probe<Matrix>* probe = new Probe<Matrix>(signal_mat, c_period);

    key_type c_key = bpy::extract<key_type>(key);
    cout << "Adding probe with key:" << c_key << endl;
    mpi_sim_chunk->add_probe(c_key, probe);
}

void PythonMpiSimulatorChunk::create_Reset(bpy::object dst, bpy::object value){
    key_type dst_key = bpy::extract<key_type>(dst);
    floattype c_value = bpy::extract<floattype>(value);

    Matrix* dst_mat = mpi_sim_chunk->get_signal(dst_key);

    Operator* reset = new Reset(dst_mat, c_value);
    mpi_sim_chunk->add_operator(reset);
}

void PythonMpiSimulatorChunk::create_Copy(bpy::object dst, bpy::object src){
    key_type dst_key = bpy::extract<key_type>(dst);
    key_type src_key = bpy::extract<key_type>(src);

    Matrix* dst_mat = mpi_sim_chunk->get_signal(dst_key);
    Matrix* src_mat = mpi_sim_chunk->get_signal(src_key);

    Operator* copy = new Copy(dst_mat, src_mat);
    mpi_sim_chunk->add_operator(copy);
}

void PythonMpiSimulatorChunk::create_DotInc(bpy::object A, bpy::object X, bpy::object Y){
    key_type A_key = bpy::extract<key_type>(A);
    key_type X_key = bpy::extract<key_type>(X);
    key_type Y_key = bpy::extract<key_type>(Y);

    Matrix* A_mat = mpi_sim_chunk->get_signal(A_key);
    Matrix* X_mat = mpi_sim_chunk->get_signal(X_key);
    Matrix* Y_mat = mpi_sim_chunk->get_signal(Y_key);

    Operator* dot_inc = new DotInc(A_mat, X_mat, Y_mat);
    mpi_sim_chunk->add_operator(dot_inc);
}

void PythonMpiSimulatorChunk::create_ElementwiseInc(bpy::object A, bpy::object X, bpy::object Y){
    key_type A_key = bpy::extract<key_type>(A);
    key_type X_key = bpy::extract<key_type>(X);
    key_type Y_key = bpy::extract<key_type>(Y);

    Matrix* A_mat = mpi_sim_chunk->get_signal(A_key);
    Matrix* X_mat = mpi_sim_chunk->get_signal(X_key);
    Matrix* Y_mat = mpi_sim_chunk->get_signal(Y_key);

    Operator* dot_inc = new ElementwiseInc(A_mat, X_mat, Y_mat);
    mpi_sim_chunk->add_operator(dot_inc);
}

void PythonMpiSimulatorChunk::create_Synapse(bpy::object input, bpy::object output,
                                            bpy::list numer, bpy::list denom){

    key_type input_key = bpy::extract<key_type>(input);
    key_type output_key = bpy::extract<key_type>(output);

    Matrix* input_mat = mpi_sim_chunk->get_signal(input_key);
    Matrix* output_mat = mpi_sim_chunk->get_signal(output_key);

    Matrix* numer_mat = list_to_matrix(numer);
    Matrix* denom_mat = list_to_matrix(denom);

    Operator* synapse = new Synapse(input_mat, output_mat, numer_mat, denom_mat);
    mpi_sim_chunk->add_operator(synapse);
}

void PythonMpiSimulatorChunk::create_SimLIF(bpy::object n_neurons, bpy::object tau_rc,
    bpy::object tau_ref, bpy::object dt, bpy::object J, bpy::object output){

    int c_n_neurons = bpy::extract<int>(n_neurons);
    floattype c_tau_rc = bpy::extract<floattype>(tau_rc);
    floattype c_tau_ref = bpy::extract<floattype>(tau_ref);
    floattype c_dt = bpy::extract<floattype>(dt);

    key_type J_key = bpy::extract<key_type>(J);
    key_type output_key = bpy::extract<key_type>(output);

    Matrix* J_mat = mpi_sim_chunk->get_signal(J_key);
    Matrix* output_mat = mpi_sim_chunk->get_signal(output_key);

    Operator* sim_lif = new SimLIF(c_n_neurons, c_tau_rc, c_tau_ref, c_dt, J_mat, output_mat);
    mpi_sim_chunk->add_operator(sim_lif);
}

void PythonMpiSimulatorChunk::create_SimLIFRate(bpy::object n_neurons, bpy::object tau_rc,
    bpy::object tau_ref, bpy::object dt, bpy::object J, bpy::object output){

    int c_n_neurons = bpy::extract<int>(n_neurons);
    floattype c_tau_rc = bpy::extract<floattype>(tau_rc);
    floattype c_tau_ref = bpy::extract<floattype>(tau_ref);
    floattype c_dt = bpy::extract<floattype>(dt);

    key_type J_key = bpy::extract<key_type>(J);
    key_type output_key = bpy::extract<key_type>(output);

    Matrix* J_mat = mpi_sim_chunk->get_signal(J_key);
    Matrix* output_mat = mpi_sim_chunk->get_signal(output_key);

    Operator* sim_lif_rate = new SimLIFRate(c_n_neurons, c_tau_rc, c_tau_ref, c_dt, J_mat, output_mat);
    mpi_sim_chunk->add_operator(sim_lif_rate);
}

void PythonMpiSimulatorChunk::create_MPISend(bpy::object dst, bpy::object tag, bpy::object content){
    int c_dst = bpy::extract<int>(dst);
    int c_tag = bpy::extract<int>(tag);
    key_type content_key = bpy::extract<key_type>(content);
    Matrix* content_matrix = mpi_sim_chunk->get_signal(content_key);

    MPISend* mpi_send = new MPISend(c_dst, c_tag, content_matrix);
    mpi_sim_chunk->add_mpi_send(mpi_send);
}

void PythonMpiSimulatorChunk::create_MPIRecv(bpy::object src, bpy::object tag, bpy::object content){
    int c_src = bpy::extract<int>(src);
    int c_tag = bpy::extract<int>(tag);
    key_type content_key = bpy::extract<key_type>(content);
    Matrix* content_matrix = mpi_sim_chunk->get_signal(content_key);

    MPIRecv* mpi_recv = new MPIRecv(c_src, c_tag, content_matrix);
    mpi_sim_chunk->add_mpi_recv(mpi_recv);
}

void PythonMpiSimulatorChunk::create_MPIWait(bpy::object tag){
    int c_tag = bpy::extract<int>(tag);

    MPIWait* mpi_wait = new MPIWait(c_tag);
    mpi_sim_chunk->add_mpi_wait(mpi_wait);
}

void PythonMpiSimulatorChunk::create_PyFunc(bpy::object py_fn, bpy::object t_in){

    bool c_t_in = bpy::extract<bool>(t_in);
    double* time_pointer = c_t_in ? mpi_sim_chunk->get_time_pointer() : NULL;

    Operator* sim_py_func = new PyFunc(NULL, py_fn, time_pointer);

    mpi_sim_chunk->add_operator(sim_py_func);
}

void PythonMpiSimulatorChunk::create_PyFuncO(bpy::object output, bpy::object py_fn, bpy::object t_in){

    key_type output_key = bpy::extract<key_type>(output);
    Matrix* output_mat = mpi_sim_chunk->get_signal(output_key);

    bool c_t_in = bpy::extract<bool>(t_in);
    double* time_pointer = c_t_in ? mpi_sim_chunk->get_time_pointer() : NULL;

    Operator* sim_py_func = new PyFunc(output_mat, py_fn, time_pointer);

    mpi_sim_chunk->add_operator(sim_py_func);
}

void PythonMpiSimulatorChunk::create_PyFuncI(
        bpy::object py_fn, bpy::object t_in, bpy::object input, bpyn::array py_input){

    key_type input_key = bpy::extract<key_type>(input);
    Matrix* input_mat = mpi_sim_chunk->get_signal(input_key);

    bool c_t_in = bpy::extract<bool>(t_in);
    double* time_pointer = c_t_in ? mpi_sim_chunk->get_time_pointer() : NULL;

    Operator* sim_py_func = new PyFunc(NULL, py_fn, time_pointer, input_mat, py_input);

    mpi_sim_chunk->add_operator(sim_py_func);
}

void PythonMpiSimulatorChunk::create_PyFuncIO(
        bpy::object output, bpy::object py_fn, bpy::object t_in,
        bpy::object input, bpyn::array py_input){

    key_type output_key = bpy::extract<key_type>(output);
    Matrix* output_mat = mpi_sim_chunk->get_signal(output_key);

    key_type input_key = bpy::extract<key_type>(input);
    Matrix* input_mat = mpi_sim_chunk->get_signal(input_key);

    bool c_t_in = bpy::extract<bool>(t_in);
    double* time_pointer = c_t_in ? mpi_sim_chunk->get_time_pointer() : NULL;

    Operator* sim_py_func = new PyFunc(
        output_mat, py_fn, time_pointer, input_mat, py_input);

    mpi_sim_chunk->add_operator(sim_py_func);
}


PyFunc::PyFunc(Matrix* output, bpy::object py_fn, double* time)
    :output(output), py_fn(py_fn), time(time), supply_time(time!=NULL), supply_input(false), input(NULL), py_input(0.0){
}

PyFunc::PyFunc(Matrix* output, bpy::object py_fn, double* time, Matrix* input, bpyn::array py_input)
    :output(output), py_fn(py_fn),  time(time), supply_time(time!=NULL), supply_input(true),
    input(input), py_input(py_input){
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
    bpy::class_<PythonMpiSimulatorChunk>("PythonMpiSimulatorChunk", bpy::init<>())
        .def("to_string", &PythonMpiSimulatorChunk::to_string)
        .def("add_signal", &PythonMpiSimulatorChunk::add_signal)
        .def("create_Probe", &PythonMpiSimulatorChunk::create_Probe)
        .def("create_Reset", &PythonMpiSimulatorChunk::create_Reset)
        .def("create_Copy", &PythonMpiSimulatorChunk::create_Copy)
        .def("create_DotInc", &PythonMpiSimulatorChunk::create_DotInc)
        .def("create_ElementwiseInc", &PythonMpiSimulatorChunk::create_ElementwiseInc)
        .def("create_Synapse", &PythonMpiSimulatorChunk::create_Synapse)
        .def("create_SimLIF", &PythonMpiSimulatorChunk::create_SimLIF)
        .def("create_SimLIFRate", &PythonMpiSimulatorChunk::create_SimLIFRate)
        .def("create_MPISend", &PythonMpiSimulatorChunk::create_MPISend)
        .def("create_MPIRecv", &PythonMpiSimulatorChunk::create_MPIRecv)
        .def("create_MPIWait", &PythonMpiSimulatorChunk::create_MPIWait)
        .def("create_PyFunc", &PythonMpiSimulatorChunk::create_PyFunc)
        .def("create_PyFuncO", &PythonMpiSimulatorChunk::create_PyFuncO)
        .def("create_PyFuncI", &PythonMpiSimulatorChunk::create_PyFuncI)
        .def("create_PyFuncIO", &PythonMpiSimulatorChunk::create_PyFuncIO);
    bpy::class_<PythonMpiSimulator>("PythonMpiSimulator", bpy::init<>())
        .def("to_string", &PythonMpiSimulator::to_string)
        .def("run_n_steps", &PythonMpiSimulator::run_n_steps)
        .def("get_probe_data", &PythonMpiSimulator::get_probe_data)
        .def("finalize", &PythonMpiSimulator::finalize)
        .def("add_chunk", &PythonMpiSimulator::add_chunk,
             bpy::return_internal_reference<>())
        .def("write_to_file", &PythonMpiSimulator::write_to_file)
        .def("read_from_file", &PythonMpiSimulator::read_from_file);
}

