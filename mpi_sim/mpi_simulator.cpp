#include "mpi_simulator.hpp"

// This constructor assumes that MPI_Initialize has already been called.
MpiSimulator::MpiSimulator()
:n_processors(0), comm(MPI_COMM_NULL){
    init();
}

MpiSimulator::MpiSimulator(int n_processors, dtype dt)
:n_processors(n_processors){
    this->dt = dt;

    spawn_processors();

    init();

    // Send blank filename, so workers know not to look for a file
    string filename;
    for(int i = 0; i < n_processors-1; i++){
        send_string(filename, i+1, setup_tag, comm);
        send_dtype(dt, i+1, setup_tag, comm);
    }
}

MpiSimulator::~MpiSimulator(){
    // TODO: send a flag to workers, telling them to clean up and exit
}

void MpiSimulator::spawn_processors(){
    int argc = 0;
    char** argv;

    cout << "Master initing MPI..." << endl;
    MPI_Init(&argc, &argv);
    cout << "Master finished initing MPI." << endl;

    cout << "Master spawning " << n_processors - 1 << " workers..." << endl;

    MPI_Comm inter;

    MPI_Comm_spawn(
        "/home/c/celiasmi/e2crawfo/nengo_mpi/nengo_mpi/mpi_sim_worker",
         MPI_ARGV_NULL, n_processors - 1,
         MPI_INFO_NULL, 0, MPI_COMM_SELF, &inter,
         MPI_ERRCODES_IGNORE);

    cout << "Master finished spawning workers." << endl;

    MPI_Intercomm_merge(inter, false, &comm);
}

void MpiSimulator::init(){
    if(comm == MPI_COMM_NULL){
        comm = MPI_COMM_WORLD;
    }

    MPI_Comm_size(comm, &n_processors);

    int buflen = 512;
    char name[buflen];
    MPI_Get_processor_name(name, &buflen);

    int rank;
    MPI_Comm_rank(comm, &rank);

    cout << "Master host: " << name << endl;
    cout << "Master rank in merged communicator: "
         << rank << " (should be 0)." << endl;
    cout << "Master detected " << n_processors << " processor(s) in total." << endl;

    chunk = shared_ptr<MpiSimulatorChunk>(new MpiSimulatorChunk(0, n_processors));

    for(int i = 0; i < n_processors; i++){
        probe_counts[i] = 0;
    }
}

void MpiSimulator::from_file(string filename){
    clock_t begin = clock();

    if(filename.length() == 0){
        stringstream s;
        s << "Got empty string for filename" << endl;
        throw runtime_error(s.str());
    }

    ifstream in_file(filename);

    if(!in_file.good()){
        stringstream s;
        s << "The network file " << filename << " does not exist." << endl;
        throw runtime_error(s.str());
    }

    in_file.close();

    for(int i = 0; i < n_processors-1; i++){
        send_string(filename, i+1, setup_tag, comm);
    }

    // Use parallel property lists
    hid_t file_plist = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(file_plist, comm, MPI_INFO_NULL);

    hid_t read_plist = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(read_plist, H5FD_MPIO_INDEPENDENT);

    chunk->from_file(filename, file_plist, read_plist, comm);

    H5Pclose(file_plist);
    H5Pclose(read_plist);

    chunk->set_communicator(comm);

    // Master barrier 1
    MPI_Barrier(comm);

    clock_t end = clock();
    cout << "Loading network from file took "
         << double(end - begin) / CLOCKS_PER_SEC << " seconds." << endl;
}

void MpiSimulator::add_base_signal(
        int component, key_type key, string label, unique_ptr<BaseSignal> data){

    int processor_index = component % n_processors;

    build_dbg(
        "SIGNAL" << delim << processor_index << delim
        << key << delim << label << delim << *data);

    if(processor_index == 0){
        chunk->add_base_signal(key, label, move(data));
    }else{
        send_int(add_signal_flag, processor_index, setup_tag, comm);

        send_key(key, processor_index, setup_tag, comm);
        send_string(label, processor_index, setup_tag, comm);
        send_matrix(move(data), processor_index, setup_tag, comm);
    }
}

void MpiSimulator::add_op(int component, string op_string){

    int processor_index = component % n_processors;

    build_dbg("OP" << delim << processor_index << delim << op_string);

    if(processor_index == 0){
        chunk->add_op(op_string);
    }else{
        send_int(add_op_flag, processor_index, setup_tag, comm);
        send_string(op_string, processor_index, setup_tag, comm);
    }
}

void MpiSimulator::add_probe(
        int component, key_type probe_key, string signal_string, dtype period, string name){

    int processor_index = component % n_processors;

    probe_counts[processor_index] += 1;
    probe_data[probe_key] = vector<unique_ptr<BaseSignal>>();

    if(processor_index == 0){
        chunk->add_probe(probe_key, signal_string, period);
    }else{
        send_int(add_probe_flag, processor_index, setup_tag, comm);

        send_key(probe_key, processor_index, setup_tag, comm);
        send_string(signal_string, processor_index, setup_tag, comm);
        send_dtype(period, processor_index, setup_tag, comm);
    }
}

SignalView MpiSimulator::get_signal(string signal_string){
    return chunk->get_signal_view(signal_string);
}

void MpiSimulator::add_op(unique_ptr<Operator> op){
    chunk->add_op(move(op));
}

void MpiSimulator::finalize_build(){
    chunk->set_communicator(comm);
}

void MpiSimulator::run_n_steps(int steps, bool progress, string log_filename){
    clock_t begin = clock();

    cout << "Master sending simulation signal to " << n_processors - 1 << " workers." << endl;
    MPI_Bcast(&steps, 1, MPI_INT, 0, comm);

    cout << "Master starting simulation: " << steps << " steps." << endl;

    chunk->set_log_filename(log_filename);
    chunk->run_n_steps(steps, progress);

    // Master barrier 2
    MPI_Barrier(comm);

    if(!chunk->is_logging()){
        gather_probe_data();
    }

    // Master barrier 3
    MPI_Barrier(comm);

    chunk->close_simulation_log();
    MPI_Finalize();

    clock_t end = clock();
    cout << "Simulating " << steps << " steps took "
         << double(end - begin) / CLOCKS_PER_SEC << " seconds." << endl;
}

void MpiSimulator::gather_probe_data(){

    Simulator::gather_probe_data();

    run_dbg("Master gathering probe data from workers..." << endl);

    for(auto& pair : probe_counts){

        int chunk_index = pair.first;
        int probe_count = pair.second;

        if(chunk_index > 0){
            for(unsigned i = 0; i < probe_count; i++){
                key_type probe_key = recv_key(chunk_index, probe_tag, comm);

                run_dbg("Master receiving probe data from chunk " << chunk_index << endl
                        << "with key " << probe_key << "..." << endl);

                int data_length = recv_int(chunk_index, probe_tag, comm);

                auto& data = probe_data[probe_key];
                data.reserve(data.size() + data_length);

                for(int j = 0; j < data_length; j++){
                    unique_ptr<BaseSignal> matrix = recv_matrix(chunk_index, probe_tag, comm);

                    data.push_back(move(matrix));
                }
            }
        }
    }

    cout << "Master done gathering probe data from workers." << endl;
}

string MpiSimulator::to_string() const{
    stringstream out;

    out << "<MpiSimulator" << endl;
    out << "num components: " << n_processors << endl;

    out << "**chunk**" << endl;
    out << *chunk << endl;

    return out.str();
}

string recv_string(int src, int tag, MPI_Comm comm){
    int size;
    MPI_Status status;

    // Convention: After this call ``size'' gives the size of
    // the string, not including the c_str's terminating character
    MPI_Recv(&size, 1, MPI_INT, src, tag, comm, &status);

    unique_ptr<char[]> buffer(new char[size+1]);

    MPI_Recv(buffer.get(), size+1, MPI_CHAR, src, tag, comm, &status);

    string s(buffer.get());

    return s;
}

void send_string(string s, int dst, int tag, MPI_Comm comm){
    int size = s.length();

    // Convention: send the size of the string, ommitting
    // the c_str's terminating character
    MPI_Send(&size, 1, MPI_INT, dst, tag, comm);

    unique_ptr<char[]> buffer(new char[size+1]);

    strcpy(buffer.get(), s.c_str());
    MPI_Send(buffer.get(), size+1, MPI_CHAR, dst, tag, comm);
}

dtype recv_dtype(int src, int tag, MPI_Comm comm){
    MPI_Status status;
    double d;

    MPI_Recv(&d, 1, MPI_DOUBLE, src, tag, comm, &status);
    return d;
}

void send_dtype(dtype d, int dst, int tag, MPI_Comm comm){
    MPI_Send(&d, 1, MPI_DOUBLE, dst, tag, comm);
}

int recv_int(int src, int tag, MPI_Comm comm){
    MPI_Status status;
    int i;

    MPI_Recv(&i, 1, MPI_INT, src, tag, comm, &status);
    return i;
}

void send_int(int i, int dst, int tag, MPI_Comm comm){
    MPI_Send(&i, 1, MPI_INT, dst, tag, comm);
}

key_type recv_key(int src, int tag, MPI_Comm comm){
    MPI_Status status;
    key_type i;

    MPI_Recv(&i, 1, MPI_LONG_LONG_INT, src, tag, comm, &status);
    return i;
}

void send_key(key_type i, int dst, int tag, MPI_Comm comm){
    MPI_Send(&i, 1, MPI_LONG_LONG_INT, dst, tag, comm);
}

unique_ptr<BaseSignal> recv_matrix(int src, int tag, MPI_Comm comm){
    MPI_Status status;

    int size1 = recv_int(src, tag, comm);
    int size2 = recv_int(src, tag, comm);

    unique_ptr<dtype[]> data_buffer(new dtype[size1 * size2]);
    MPI_Recv(data_buffer.get(), size1 * size2, MPI_DOUBLE, src, tag, comm, &status);

    auto matrix = unique_ptr<BaseSignal>(new BaseSignal(size1, size2));

    // Assumes matrices stored in row-major
    for(int i = 0; i < size1; i++){
        for(int j = 0; j < size2; j++){
            (*matrix)(i, j) = data_buffer[i * size2 + j];
        }
    }

    return move(matrix);
}

void send_matrix(unique_ptr<BaseSignal> matrix, int dst, int tag, MPI_Comm comm){

    int size1 = matrix->size1(), size2 = matrix->size2();
    send_int(size1, dst, tag, comm);
    send_int(size2, dst, tag, comm);

    unique_ptr<dtype[]> data_buffer(new dtype[size1 * size2]);

    // Assumes matrices stored in row-major
    for(int i = 0; i < size1; i++){
        for(int j = 0; j < size2; j++){
            data_buffer[i * size2 + j] = (*matrix)(i, j) ;
        }
    }

    MPI_Send(data_buffer.get(), size1 * size2, MPI_DOUBLE, dst, tag, comm);
}