#include "mpi_simulator.hpp"

int n_processors_available = 1;

// This constructor assumes that MPI_Initialize has already been called.
MpiSimulator::MpiSimulator(bool mpi_merged, bool collect_timings)
:Simulator(collect_timings), comm(MPI_COMM_WORLD), mpi_merged(mpi_merged){
    MPI_Comm_size(comm, &n_processors);

    int buflen = 512;
    char name[buflen];
    MPI_Get_processor_name(name, &buflen);

    int rank;
    MPI_Comm_rank(comm, &rank);

    cout << "Master host: " << name << endl;
    cout << "Master rank in merged communicator: " << rank << " (should be 0)." << endl;
    cout << "Master detected " << n_processors << " processor(s) in total." << endl;

    wake_workers();
    bcast_send_int(mpi_merged ? 1 : 0, comm);
    bcast_send_int(collect_timings ? 1 : 0, comm);

    chunk = unique_ptr<MpiSimulatorChunk>(
        new MpiSimulatorChunk(0, n_processors, mpi_merged, collect_timings));
}

MpiSimulator::~MpiSimulator(){
    // TODO: send a flag to workers, telling them to clean up and exit
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

    chunk->from_file(filename, file_plist, read_plist);

    probe_counts.resize(n_processors);
    for(const ProbeSpec& pi : chunk->probe_info){
        probe_data[pi.probe_key] = vector<unique_ptr<BaseSignal>>();
        probe_counts[pi.component % n_processors] += 1;
    }

    H5Pclose(file_plist);
    H5Pclose(read_plist);

    // Master barrier 1
    MPI_Barrier(comm);

    clock_t end = clock();
    cout << "Loading network from file took "
         << double(end - begin) / CLOCKS_PER_SEC << " seconds." << endl;
}

void MpiSimulator::finalize_build(){
    chunk->finalize_build(comm);
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

    clock_t end = clock();
    cout << "Simulating " << steps << " steps took "
         << double(end - begin) / CLOCKS_PER_SEC << " seconds." << endl;
}

void MpiSimulator::gather_probe_data(){

    // Gather data on the master process
    Simulator::gather_probe_data();

    // Gather data on the worker processes
    cout << "Master gathering probe data from workers..." << endl;

    for(int processor_idx = 1; processor_idx < n_processors; processor_idx++){

        int probe_count = probe_counts[processor_idx];

        for(unsigned i = 0; i < probe_count; i++){
            key_type probe_key = recv_key(processor_idx, probe_tag, comm);

            run_dbg("Master receiving probe data from chunk " << processor_idx << endl
                    << "with key " << probe_key << "..." << endl);

            int data_length = recv_int(processor_idx, probe_tag, comm);

            auto& data = probe_data[probe_key];
            data.reserve(data.size() + data_length);

            for(int j = 0; j < data_length; j++){
                unique_ptr<BaseSignal> matrix = recv_matrix(processor_idx, probe_tag, comm);

                data.push_back(move(matrix));
            }
        }
    }

    cout << "Master done gathering probe data from workers." << endl;
}

void MpiSimulator::close(){
    cout << "Master sending termination signal to " << n_processors - 1 << " workers." << endl;
    int signal = -1;
    MPI_Bcast(&signal, 1, MPI_INT, 0, comm);

    // Master barrier 4
    MPI_Barrier(comm);

    chunk->close_simulation_log();
}

string MpiSimulator::to_string() const{
    stringstream out;

    out << "<MpiSimulator" << endl;
    out << "n_processors: " << n_processors << endl;

    out << "**chunk**" << endl;
    out << *chunk << endl;

    return out.str();
}

void mpi_init(){
    int argc = 0;
    char** argv;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &n_processors_available);
}

void mpi_finalize(){
    MPI_Finalize();
}

int get_mpi_rank(){
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
};

int get_mpi_n_procs(){
    return n_processors_available;
}

void wake_workers(){
    int kill = 0;
    MPI_Bcast(&kill, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

void kill_workers(){
    int kill = 1;
    MPI_Bcast(&kill, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

void worker_start(){
    worker_start(MPI_COMM_WORLD);
}

// comm: The communicator for the worker to communicate on. Must
// be an intracommunicator involving all processes, with the master
// process having rank 0.
void worker_start(MPI_Comm comm){

    int rank, n_processors;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &n_processors);

    int buflen = 512;
    char name[buflen];
    MPI_Get_processor_name(name, &buflen);

    dbg("Hello world! I'm a nengo_mpi worker process with "
        "rank "<< rank << " on host " << name << "." << endl);

    while(true){
        int kill;
        MPI_Bcast(&kill, 1, MPI_INT, 0, comm);

        if(kill){
            // Program is ending
            break;
        }

        MPI_Status status;

        dbg("Reading merged...");
        int mpi_merged = bcast_recv_int(comm);

        dbg("Reading collect_timings...");
        int collect_timings = bcast_recv_int(comm);

        dbg("Reading filename...");
        string filename = recv_string(0, setup_tag, comm);

        dbg("Creating chunk...");
        MpiSimulatorChunk chunk(rank, n_processors, bool(mpi_merged), bool(collect_timings));

        // Use parallel property lists
        hid_t file_plist = H5Pcreate(H5P_FILE_ACCESS);
        H5Pset_fapl_mpio(file_plist, comm, MPI_INFO_NULL);

        hid_t read_plist = H5Pcreate(H5P_DATASET_XFER);
        H5Pset_dxpl_mpio(read_plist, H5FD_MPIO_INDEPENDENT);

        dbg("Loading from file...");
        chunk.from_file(filename, file_plist, read_plist);
        chunk.finalize_build(comm);

        H5Pclose(file_plist);
        H5Pclose(read_plist);

        // Worker barrier 1
        MPI_Barrier(comm);

        while(true){
            dbg("Worker " << rank << " waiting for signal to start simulation...");
            int steps;
            MPI_Bcast(&steps, 1, MPI_INT, 0, comm);

            if(steps < 0){
                // Current simulator is closing
                break;
            }

            dbg("Worker " << rank << " received the signal to start simulation: "
                << steps << " steps." << endl);

            chunk.run_n_steps(steps, false);

            // Worker barrier 2
            MPI_Barrier(comm);

            if(!chunk.is_logging()){
                // If we're not logging, send the probe data back to the master
                for(auto& pair : chunk.probe_map){
                    key_type key = pair.first;
                    shared_ptr<Probe>& probe = pair.second;

                    send_key(key, 0, probe_tag, comm);

                    vector<unique_ptr<BaseSignal>> probe_data = probe->harvest_data();

                    send_int(probe_data.size(), 0, probe_tag, comm);

                    for(auto& pd : probe_data){
                        send_matrix(move(pd), 0, probe_tag, comm);
                    }
                }
            }

            // Worker barrier 3
            MPI_Barrier(comm);
        }

        dbg("Worker " << rank << " received the signal to terminate.");

        // Worker barrier 4
        MPI_Barrier(comm);

        chunk.close_simulation_log();
    }
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

int bcast_recv_int(MPI_Comm comm){
    int src = 0;

    int i;
    MPI_Bcast(&i, 1, MPI_INT, src, comm);
    return i;
}

void bcast_send_int(int i, MPI_Comm comm){
    int src = 0;

    MPI_Bcast(&i, 1, MPI_INT, src, comm);
}