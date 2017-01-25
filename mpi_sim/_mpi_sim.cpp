#include "_mpi_sim.hpp"

static char module_docstring[] = "TODO";

static char init_docstring[] = "TODO";
static char finalize_docstring[] = "TODO";
static char get_rank_docstring[] = "TODO";
static char get_n_procs_docstring[] = "TODO";
static char kill_workers_docstring[] = "TODO";
static char worker_start_docstring[] = "TODO";

static char create_simulator_docstring[] = "TODO";
static char load_network_docstring[] = "TODO";
static char finalize_build_docstring[] = "TODO";

static char run_n_steps_docstring[] = "TODO";
static char get_probe_data_docstring[] = "TODO";
static char get_signal_value_docstring[] = "TODO";
static char reset_simulator_docstring[] = "TODO";
static char close_simulator_docstring[] = "TODO";
static char create_PyFunc_docstring[] = "TODO";

extern "C" PyObject* mpi_sim_init(PyObject *self, PyObject *args);
extern "C" PyObject* mpi_sim_finalize(PyObject *self, PyObject *args);
extern "C" PyObject* mpi_sim_get_rank(PyObject *self, PyObject *args);
extern "C" PyObject* mpi_sim_get_n_procs(PyObject *self, PyObject *args);
extern "C" PyObject* mpi_sim_kill_workers(PyObject *self, PyObject *args);
extern "C" PyObject* mpi_sim_worker_start(PyObject *self, PyObject *args);

extern "C" PyObject* mpi_sim_create_simulator(PyObject *self, PyObject *args);
extern "C" PyObject* mpi_sim_load_network(PyObject *self, PyObject *args);
extern "C" PyObject* mpi_sim_finalize_build(PyObject *self, PyObject *args);

extern "C" PyObject* mpi_sim_run_n_steps(PyObject *self, PyObject *args);
extern "C" PyObject* mpi_sim_get_probe_data(PyObject *self, PyObject *args);
extern "C" PyObject* mpi_sim_get_signal_value(PyObject *self, PyObject *args);
extern "C" PyObject* mpi_sim_reset_simulator(PyObject *self, PyObject *args);
extern "C" PyObject* mpi_sim_close_simulator(PyObject *self, PyObject *args);
extern "C" PyObject* mpi_sim_create_PyFunc(PyObject *self, PyObject *args);

static PyMethodDef module_methods[] = {
    {"init", mpi_sim_init, METH_VARARGS, init_docstring},
    {"finalize", mpi_sim_finalize, METH_VARARGS, finalize_docstring},
    {"get_rank", mpi_sim_get_rank, METH_VARARGS, get_rank_docstring},
    {"get_n_procs", mpi_sim_get_n_procs, METH_VARARGS, get_n_procs_docstring},
    {"kill_workers", mpi_sim_kill_workers, METH_VARARGS, kill_workers_docstring},
    {"worker_start", mpi_sim_worker_start, METH_VARARGS, worker_start_docstring},

    {"create_simulator", mpi_sim_create_simulator, METH_VARARGS, create_simulator_docstring},
    {"load_network", mpi_sim_load_network, METH_VARARGS, load_network_docstring},
    {"finalize_build", mpi_sim_finalize_build, METH_VARARGS, finalize_build_docstring},

    {"run_n_steps", mpi_sim_run_n_steps, METH_VARARGS, run_n_steps_docstring},
    {"get_probe_data", mpi_sim_get_probe_data, METH_VARARGS, get_probe_data_docstring},
    {"get_signal_value", mpi_sim_get_signal_value, METH_VARARGS, get_signal_value_docstring},
    {"reset_simulator", mpi_sim_reset_simulator, METH_VARARGS, reset_simulator_docstring},
    {"close_simulator", mpi_sim_close_simulator, METH_VARARGS, close_simulator_docstring},
    {"create_PyFunc", mpi_sim_create_PyFunc, METH_VARARGS, create_PyFunc_docstring},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initmpi_sim(void)
{
    PyObject *m = Py_InitModule3("mpi_sim", module_methods, module_docstring);
    if (m == NULL)
        return;

    /* Load `numpy` functionality. */
    import_array();
}

extern "C" PyObject *mpi_sim_init(PyObject *self, PyObject *args){

    if(!PyArg_ParseTuple(args, "")){
        return NULL;
    }

    mpi_init();

    Py_INCREF(Py_None);
    return Py_None;
}

extern "C" PyObject *mpi_sim_finalize(PyObject *self, PyObject *args){
    if(!PyArg_ParseTuple(args, "")){
        return NULL;
    }

    mpi_finalize();

    Py_INCREF(Py_None);
    return Py_None;
}

extern "C" PyObject *mpi_sim_get_rank(PyObject *self, PyObject *args){
    if(!PyArg_ParseTuple(args, "")){
        return NULL;
    }

    int rank = mpi_get_rank();
    return Py_BuildValue("i", rank);
}

extern "C" PyObject *mpi_sim_get_n_procs(PyObject *self, PyObject *args){
    if(!PyArg_ParseTuple(args, "")){
        return NULL;
    }

    int n_procs = mpi_get_n_procs();
    return Py_BuildValue("i", n_procs);
}

extern "C" PyObject *mpi_sim_kill_workers(PyObject *self, PyObject *args){
    if(!PyArg_ParseTuple(args, "")){
        return NULL;
    }

    mpi_kill_workers();

    Py_INCREF(Py_None);
    return Py_None;
}

extern "C" PyObject *mpi_sim_worker_start(PyObject *self, PyObject *args){
    if(!PyArg_ParseTuple(args, "")){
        return NULL;
    }

    mpi_worker_start();

    Py_INCREF(Py_None);
    return Py_None;
}

unique_ptr<Simulator> simulator;

extern "C" PyObject *mpi_sim_create_simulator(PyObject *self, PyObject *args){
    if(!PyArg_ParseTuple(args, "")){
        return NULL;
    }

    if(n_processors_available == 1){
        simulator = unique_ptr<Simulator>(new Simulator(false));
    }else{
        simulator = unique_ptr<Simulator>(new MpiSimulator(false));
    }

    Py_INCREF(Py_None);
    return Py_None;
}

extern "C" PyObject *mpi_sim_load_network(PyObject *self, PyObject *args){
    const char *filename;
    if(!PyArg_ParseTuple(args, "s", &filename)){
        return NULL;
    }

    simulator->from_file(filename);

    Py_INCREF(Py_None);
    return Py_None;
}

extern "C" PyObject *mpi_sim_finalize_build(PyObject *self, PyObject *args){
    if(!PyArg_ParseTuple(args, "")){
        return NULL;
    }

    simulator->finalize_build();

    Py_INCREF(Py_None);
    return Py_None;
}

extern "C" PyObject *mpi_sim_run_n_steps(PyObject *self, PyObject *args){
    int n_steps, progress;
    const char *log_filename;
    if(!PyArg_ParseTuple(args, "iis", &n_steps, &progress, &log_filename)){
        return NULL;
    }

    simulator->run_n_steps(n_steps, progress, log_filename);

    Py_INCREF(Py_None);
    return Py_None;
}

extern "C" PyObject *mpi_sim_get_probe_data(PyObject *self, PyObject *args){
    key_type probe_key;
    if(!PyArg_ParseTuple(args, "L", &probe_key)){
        return NULL;
    }

    vector<Signal> data = simulator->get_probe_data(probe_key);

    PyObject *lst = PyList_New(0);
    if(lst == NULL){
        return NULL;
    }

    PyObject *array;
    npy_intp shape[2];
    int ndim, success;

    for (auto& d: data){
        shape[0] = d.shape1;
        if(d.ndim == 1){
            shape[1] = 0;
            ndim = 1;
        }else{
            shape[1] = d.shape2;
            ndim = 2;
        }

        array = PyArray_SimpleNew(ndim, shape, NPY_DOUBLE);
        if (array == NULL) return NULL; // TODO
        d.copy_to_buffer((dtype*)(PyArray_DATA((PyArrayObject*)(array))));

        success = PyList_Append(lst, array);
        if(success < 0) return NULL; // TODO
    }

    return lst;
}

extern "C" PyObject *mpi_sim_get_signal_value(PyObject *self, PyObject *args){
    key_type signal_key;
    if(!PyArg_ParseTuple(args, "L", &signal_key)){
        return NULL;
    }

    Signal signal = simulator->get_signal(signal_key);
    PyObject *array;
    npy_intp shape[2];
    int ndim, success;

    shape[0] = signal.shape1;
    if(signal.ndim == 1){
        shape[1] = 0;
        ndim = 1;
    }else{
        shape[1] = signal.shape2;
        ndim = 2;
    }

    array = PyArray_SimpleNew(ndim, shape, NPY_DOUBLE);
    if (array == NULL) return NULL; // TODO
    signal.copy_to_buffer((dtype*)(PyArray_DATA((PyArrayObject*)(array))));

    return array;
}

extern "C" PyObject *mpi_sim_reset_simulator(PyObject *self, PyObject *args){
    unsigned seed;
    if(!PyArg_ParseTuple(args, "I", &seed)){
        return NULL;
    }

    simulator->reset(seed);

    Py_INCREF(Py_None);
    return Py_None;
}

extern "C" PyObject *mpi_sim_close_simulator(PyObject *self, PyObject *args){
    if(!PyArg_ParseTuple(args, "")){
        return NULL;
    }

    simulator->close();

    Py_INCREF(Py_None);
    return Py_None;
}

extern "C" PyObject *mpi_sim_create_PyFunc(PyObject *self, PyObject *args){
    PyObject *callback;
    char *time_string, *input_string, *output_string;
    PyArrayObject *py_time_buffer, *py_input_buffer, *py_output_buffer;
    float index;

    if(!PyArg_ParseTuple(args, "OsssOOOf", &callback, &time_string, &input_string, &output_string,
                         &py_time_buffer, &py_input_buffer, &py_output_buffer, &index)){
        return NULL;
    }

    if (!PyCallable_Check(callback)) {
        PyErr_SetString(PyExc_TypeError, "Parameter ``callback`` must be callable.");
        return NULL;
    }
    Py_INCREF(callback);

    Signal time = simulator->get_signal_view(time_string);
    build_dbg("Time signal: " << time);

    Signal input = simulator->get_signal_view(input_string);
    build_dbg("Input signal: " << input);

    Signal output = simulator->get_signal_view(output_string);
    build_dbg("Output signal: " << output);

    dtype* time_buffer = (dtype*)(PyArray_DATA(py_time_buffer));
    dtype* input_buffer = (dtype*)(PyArray_DATA(py_input_buffer));
    dtype* output_buffer = (dtype*)(PyArray_DATA(py_output_buffer));

    auto pyfunc = unique_ptr<Operator>(
        new PyFunc(callback, time, input, output, time_buffer, input_buffer, output_buffer));

    simulator->add_pyfunc(index, move(pyfunc));

    Py_INCREF(Py_None);
    return Py_None;
}

PyFunc::PyFunc(
    PyObject* fn, Signal time, Signal input, Signal output,
    dtype* time_buffer, dtype* input_buffer, dtype* output_buffer)
:fn(fn), time(time), input(input), output(output),
time_buffer(time_buffer), input_buffer(input_buffer), output_buffer(output_buffer){
}

void PyFunc::operator() (){
    // TODO: currently assuming pyfuncs only accept and return vectors.
    for(unsigned i = 0; i < time.shape1; i++){
        time_buffer[i] = time(i);
    }

    for(unsigned i = 0; i < input.shape1; i++){
        input_buffer[i] = input(i);
    }

    PyObject* arglist = Py_BuildValue("()");
    PyObject* result = PyObject_CallObject(fn, arglist);
    Py_DECREF(arglist);
    if(result == NULL){
        throw PythonException();
    }

    for(unsigned i = 0; i < output.shape1; i++){
        output(i) = output_buffer[i];
    }

    run_dbg(*this);
}

PyFunc::~PyFunc(){
    Py_XDECREF(fn);
}

string PyFunc::to_string() const{
    stringstream out;

    out << "PyFunc: " << endl;
    out << "Time: " << endl;
    out << time << endl << endl;
    out << "Input: " << endl;
    out << input << endl << endl;
    out << "Output: " << endl;
    out << output << endl << endl;

    out << "time_buffer: " << time_buffer << endl;
    for(int i = 0; i < time.shape1; i++){
        out << time_buffer[i] << ",";
    }
    out << endl;

    out << "input_buffer: " << input_buffer << endl;
    for(int i = 0; i < input.shape1; i++){
        out << input_buffer[i] << ",";
    }
    out << endl;

    out << "output_buffer: " << output_buffer << endl;
    for(int i = 0; i < output.shape1; i++){
        out << output_buffer[i] << ",";
    }
    out << endl;

    return out.str();
}
