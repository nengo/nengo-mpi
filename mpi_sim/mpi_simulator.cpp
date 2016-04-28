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
}

void MpiSimulator::from_file(string filename){
    clock_t begin = clock();

    label = filename;
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
        probe_data[pi.probe_key] = vector<Signal>();
        probe_counts[pi.component % n_processors] += 1;
    }

    H5Pclose(file_plist);
    H5Pclose(read_plist);

    // Master barrier 1
    MPI_Barrier(comm);

    clock_t end = clock();
    double delta = double(end - begin) / CLOCKS_PER_SEC;
    cout << "Loading network from file took " << delta << " seconds." << endl;

    write_to_loadtimes_file(delta);
}

void MpiSimulator::finalize_build(){
    chunk->finalize_build(comm);
}

void MpiSimulator::run_n_steps(int steps, bool progress, string log_filename){
    clock_t begin = clock();

    if(steps <= 0){
        throw runtime_error("Number of steps must be > 0.");
    }

    cout << "Master sending signal to run the simulation for " << steps
         << " to " << n_processors - 1 << " workers." << endl;
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
    double delta = double(end - begin) / CLOCKS_PER_SEC;
    cout << "Simulating " << steps << " steps took " << delta << " seconds." << endl;

    write_to_runtimes_file(delta);
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
                Signal signal = recv_base_signal(processor_idx, probe_tag, comm);
                data.push_back(signal);
            }
        }
    }

    cout << "Master done gathering probe data from workers." << endl;
}

void MpiSimulator::reset(unsigned seed){
    cout << "Master sending signal to reset the simulator to " << n_processors - 1 << " workers." << endl;

    int steps = 0;
    MPI_Bcast(&steps, 1, MPI_INT, 0, comm);

    // Reset
    MPI_Bcast(&seed, 1, MPI_UNSIGNED, 0, comm);
    Simulator::reset(seed);

    // Master barrier 5
    MPI_Barrier(comm);
}

void MpiSimulator::close(){
    cout << "Master sending signal to close the simulator to " << n_processors - 1 << " workers." << endl;
    int steps = -1;
    MPI_Bcast(&steps, 1, MPI_INT, 0, comm);

    chunk->close_simulation_log();

    // Master barrier 4
    MPI_Barrier(comm);
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

    // Loops once per nengo_mpi.Simulator created
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

            if(steps == 0){
                // Reset
                dbg("Worker " << rank << " received the signal to reset the simulation." << endl);

                unsigned seed;
                MPI_Bcast(&seed, 1, MPI_UNSIGNED, 0, comm);

                chunk.reset(seed);

                // Worker barrier 5
                MPI_Barrier(comm);

            }else if(steps > 0){
                // Simulate
                dbg("Worker " << rank << " received the signal to run the simulation for "
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

                        vector<Signal> probe_data = probe->harvest_data();

                        send_int(probe_data.size(), 0, probe_tag, comm);

                        for(auto& pd : probe_data){
                            send_base_signal(move(pd), 0, probe_tag, comm);
                        }
                    }
                }

                // Worker barrier 3
                MPI_Barrier(comm);
            }else{
                dbg("Worker " << rank << " received the signal to close the simulation." << endl);

                chunk.close_simulation_log();

                // Worker barrier 4
                MPI_Barrier(comm);
                break;
            }
        }
    }
}

void MpiSimulator::write_to_time_file(char* filename, double delta){
    if(filename){
        ofstream f(filename, ios::app);

        if(f.good()){
            if(f.tellp() == 0){
                // Write the header
                f << "seconds,nprocs,label" << endl;
            }

            f << delta << "," << n_processors << "," << label << endl;
        }

        f.close();
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

unsigned recv_unsigned(int src, int tag, MPI_Comm comm){
    MPI_Status status;
    unsigned i;

    MPI_Recv(&i, 1, MPI_UNSIGNED, src, tag, comm, &status);
    return i;
}

void send_unsigned(unsigned i, int dst, int tag, MPI_Comm comm){
    MPI_Send(&i, 1, MPI_UNSIGNED, dst, tag, comm);
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

Signal recv_base_signal(int src, int tag, MPI_Comm comm){
    MPI_Status status;

    unsigned size1 = recv_unsigned(src, tag, comm);
    unsigned size2 = recv_unsigned(src, tag, comm);

    Signal signal = Signal(size1, size2);
    MPI_Recv(signal.raw_data, signal.size, MPI_DOUBLE, src, tag, comm, &status);

    return signal;
}

void send_base_signal(Signal signal, int dst, int tag, MPI_Comm comm){

    send_unsigned(signal.shape1, dst, tag, comm);
    send_unsigned(signal.shape2, dst, tag, comm);

    MPI_Send(signal.raw_data, signal.size, MPI_DOUBLE, dst, tag, comm);
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

unsigned bcast_recv_unsigned(MPI_Comm comm){
    int src = 0;

    unsigned i;
    MPI_Bcast(&i, 1, MPI_UNSIGNED, src, comm);
    return i;
}

void bcast_send_unsigned(unsigned i, MPI_Comm comm){
    int src = 0;

    MPI_Bcast(&i, 1, MPI_UNSIGNED, src, comm);
}