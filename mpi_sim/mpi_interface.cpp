#include "mpi_interface.hpp"

string recv_string(int src, int tag, MPI_Comm comm){
    int size;
    MPI_Status status;

    // Convention: recv the size of the string, ommitting
    // the c_str's terminating character
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

void MpiInterface::initialize_chunks(
        bool spawn, shared_ptr<MpiSimulatorChunk> chunk, int num_chunks){

    master_chunk = chunk;
    num_remote_chunks = num_chunks;

    int argc = 0;
    char** argv;

    if(spawn){

        cout << "Master initing MPI..." << endl;
        MPI_Init(&argc, &argv);
        cout << "Master finished initing MPI." << endl;

        cout << "Master spawning " << num_remote_chunks << " children..." << endl;

        MPI_Comm inter;

        MPI_Comm_spawn(
            "/home/c/celiasmi/e2crawfo/nengo_mpi/nengo_mpi/mpi_sim_worker",
             MPI_ARGV_NULL, num_remote_chunks,
             MPI_INFO_NULL, 0, MPI_COMM_SELF, &inter,
             MPI_ERRCODES_IGNORE);

        cout << "Master finished spawning children." << endl;

        MPI_Intercomm_merge(inter, false, &comm);
    }

    int buflen = 512;
    char name[buflen];
    MPI_Get_processor_name(name, &buflen);

    int rank;
    MPI_Comm_rank(comm, &rank);

    cout << "Master host: " << name << endl;
    cout << "Master rank in merged communicator: "
         << rank << " (should be 0)." << endl;

    dtype dt = master_chunk->dt;
    string chunk_label;

    for(int i = 0; i < num_remote_chunks; i++){
        stringstream s;
        s << "Chunk " << i + 1;
        chunk_label = s.str();

        send_string(chunk_label, i+1, setup_tag, comm);
        send_dtype(dt, i+1, setup_tag, comm);
    }
}

void MpiInterface::add_base_signal(
        int component, key_type key, string label, unique_ptr<BaseSignal> data){

    send_int(add_signal_flag, component, setup_tag, comm);

    send_key(key, component, setup_tag, comm);
    send_string(label, component, setup_tag, comm);
    send_matrix(move(data), component, setup_tag, comm);
}

void MpiInterface::add_op(int component, string op_string){
    send_int(add_op_flag, component, setup_tag, comm);

    send_string(op_string, component, setup_tag, comm);
}

void MpiInterface::add_probe(
        int component, key_type probe_key, string signal_string, dtype period){

    send_int(add_probe_flag, component, setup_tag, comm);

    send_key(probe_key, component, setup_tag, comm);
    send_string(signal_string, component, setup_tag, comm);
    send_dtype(period, component, setup_tag, comm);
}

void MpiInterface::finalize(){

    master_chunk->set_communicator(comm);
    master_chunk->add_op(unique_ptr<Operator>(new MPIBarrier(comm)));

    for(int i = 0; i < num_remote_chunks; i++){
        send_int(stop_flag, i+1, setup_tag, comm);
    }
}

void MpiInterface::run_n_steps(int steps, bool progress){
    cout << "Master sending simulation signal." << endl;
    MPI_Bcast(&steps, 1, MPI_INT, 0, comm);

    cout << "Master starting simulation: " << steps << " steps." << endl;

    master_chunk->run_n_steps(steps, progress);

    MPI_Barrier(comm);
}

void MpiInterface::gather_probe_data(
        map<key_type, vector<unique_ptr<BaseSignal>>>& probe_data,
        map<int, int>& probe_counts){

    cout << "Master gathering probe data from children..." << endl;

    for(auto& pair : probe_counts){

        int chunk_index = pair.first;
        int probe_count = pair.second;

        if(chunk_index > 0){
            for(unsigned i = 0; i < probe_count; i++){
                key_type probe_key = recv_key(chunk_index, probe_tag, comm);

                run_dbg("Master receiving probe from chunk " << chunk_index << endl
                        <<" with key " << probe_key << "..." << endl);

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

    cout << "Master done gathering probe data from children." << endl;

    MPI_Barrier(comm);
}

void MpiInterface::finish_simulation(){
    MPI_Finalize();
}