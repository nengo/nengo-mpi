#include "python.hpp"

void init(){
    mpi_init();
}

void finalize(){
    mpi_finalize();
}

int get_rank(){
    return mpi_get_rank();
}

int get_n_procs(){
    return mpi_get_n_procs();
}

void kill_workers(){
    return mpi_kill_workers();
}

void worker_start(){
    mpi_worker_start();
}

unique_ptr<Simulator> simulator;

void create_simulator(){
    if(n_processors_available == 1){
        simulator = unique_ptr<Simulator>(new Simulator(false));
    }else{
        simulator = unique_ptr<Simulator>(new MpiSimulator(false));
    }
}

void load_network(char* filename){
    simulator->from_file(filename);
}

void finalize_build(){
    simulator->finalize_build();
}

void run_n_steps(int n_steps, int progress, char* log_filename){
    simulator->run_n_steps(n_steps, progress, log_filename);
}

// Result of this call needs to be freed once the caller is done with it.
dtype* get_probe_data(key_type probe_key, size_t* n_signals, size_t* signal_size){
    vector<Signal> data = simulator->get_probe_data(probe_key);

    *n_signals = data.size();

    if(*n_signals > 0){
        *signal_size = data[0].size;
    }else{
        *signal_size = 0;
    }

    size_t size = (*n_signals) * (*signal_size);

    dtype* output = (dtype*) malloc(size * sizeof(dtype));
    dtype* offset = output;

    for (auto& d: data){
        d.copy_to_buffer(offset);
        offset += d.size;
    }

    return output;
}

dtype* get_signal_value(key_type key, size_t* shape1, size_t* shape2){
    Signal signal = simulator->get_signal(key);
    *shape1 = signal.shape1;
    *shape2 = signal.shape2;

    dtype* output = (dtype*) malloc(signal.size * sizeof(dtype));
    int offset = 0;
    for(unsigned i=0; i < signal.shape1; i++){
        for(unsigned j=0; j < signal.shape2; j++){
            output[offset++] = signal(i, j);
        }
    }

    return output;
}

void free_ptr(dtype* ptr){
    free(ptr);
}

void reset_simulator(unsigned seed){
    simulator->reset(seed);
}

void close_simulator(){
    simulator->close();
}

void create_PyFunc(
        py_func_t py_fn, char* time_string, char* input_string, char* output_string,
        dtype* py_time, dtype* py_input, dtype* py_output, float index){

    Signal time = simulator->get_signal_view(time_string);
    build_dbg("Time signal: " << time);

    Signal input = simulator->get_signal_view(input_string);
    build_dbg("Input signal: " << input);

    Signal output = simulator->get_signal_view(output_string);
    build_dbg("Output signal: " << output);

    auto pyfunc = unique_ptr<Operator>(
        new PyFunc(py_fn, time, input, output, py_time, py_input, py_output));

    simulator->add_pyfunc(index, move(pyfunc));
}

PyFunc::PyFunc(
    py_func_t py_fn, Signal time, Signal input, Signal output,
    dtype* py_time, dtype* py_input, dtype* py_output)
:py_fn(py_fn), time(time), input(input), output(output),
py_time(py_time), py_input(py_input), py_output(py_output){
}

void PyFunc::operator() (){
    // TODO: currently assuming pyfuncs only accept and return vectors.
    for(unsigned i = 0; i < time.shape1; i++){
        py_time[i] = time(i);
    }

    for(unsigned i = 0; i < input.shape1; i++){
        py_input[i] = input(i);
    }

    py_fn();

    for(unsigned i = 0; i < output.shape1; i++){
        output(i) = py_output[i];
    }

    run_dbg(*this);
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

    out << "py_time: " << py_time << endl;
    for(int i = 0; i < time.shape1; i++){
        out << py_time[i] << ",";
    }
    out << endl;

    out << "py_input: " << py_input << endl;
    for(int i = 0; i < input.shape1; i++){
        out << py_input[i] << ",";
    }
    out << endl;

    out << "py_output: " << py_output << endl;
    for(int i = 0; i < output.shape1; i++){
        out << py_output[i] << ",";
    }
    out << endl;

    return out.str();
}
