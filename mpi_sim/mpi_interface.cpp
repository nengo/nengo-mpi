#include "mpi_interface.hpp"

string recv_string(int src, int tag, MPI_Comm comm){
    int size;
    MPI_Status status;

    // Convention: recv the size of the string, ommitting the c_str's terminating character
    MPI_Recv(&size, 1, MPI_INT, src, tag, comm, &status);

    char* buffer = new char[size+1];
    MPI_Recv(buffer, size+1, MPI_CHAR, src, tag, comm, &status);

    string s(buffer);
    free(buffer);

    return s;
}

void send_string(string s, int dst, int tag, MPI_Comm comm){
    int size = s.length();

    // Convention: send the size of the string, ommitting the c_str's terminating character
    MPI_Send(&size, 1, MPI_INT, dst, tag, comm);

    char* buffer = new char[size+1];
    strcpy(buffer, s.c_str());
    MPI_Send(buffer, size+1, MPI_CHAR, dst, tag, comm);

    free(buffer);
}

float recv_float(int src, int tag, MPI_Comm comm){
    MPI_Status status;

    float f;
    MPI_Recv(&f, 1, MPI_FLOAT, src, tag, comm, &status);
    return f;
}

void send_float(float f, int dst, int tag, MPI_Comm comm){
    MPI_Send(&f, 1, MPI_FLOAT, dst, tag, comm);
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

BaseMatrix* recv_matrix(int src, int tag, MPI_Comm comm){
    MPI_Status status;

    int size1 = recv_int(src, tag, comm);
    int size2 = recv_int(src, tag, comm);

    floattype* data_buffer = new floattype[size1 * size2];
    MPI_Recv(data_buffer, size1 * size2, MPI_DOUBLE, src, tag, comm, &status);

    BaseMatrix* matrix = new BaseMatrix(size1, size2);

    // Assumes matrices stored in row-major
    for(int i = 0; i < size1; i++){
        for(int j = 0; j < size2; j++){
            (*matrix)(i, j) = data_buffer[i * size2 + j];
        }
    }

    free(data_buffer);

    return matrix;
}

void send_matrix(BaseMatrix* matrix, int dst, int tag, MPI_Comm comm){
    int size1 = matrix->size1(), size2 = matrix->size2();
    send_int(size1, dst, tag, comm);
    send_int(size2, dst, tag, comm);

    floattype* data_buffer = new floattype[size1 * size2];

    // Assumes matrices stored in row-major
    for(int i = 0; i < size1; i++){
        for(int j = 0; j < size2; j++){
            data_buffer[i * size2 + j] = (*matrix)(i, j) ;
        }
    }

    MPI_Send(data_buffer, size1 * size2, MPI_DOUBLE, dst, tag, comm);

    free(data_buffer);
}

void MpiInterface::initialize_chunks(bool spawn, MpiSimulatorChunk* chunk, int num_chunks){
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
    cout << "Master rank in merged communicator: " << rank << " (should be 0)." << endl;

    float dt = master_chunk->dt;
    string chunk_label;

    for(int i = 0; i < num_remote_chunks; i++){
        stringstream s;
        s << "Chunk " << i + 1;
        chunk_label = s.str();

        send_string(chunk_label, i+1, setup_tag, comm);
        send_float(dt, i+1, setup_tag, comm);
    }
}

void MpiInterface::add_base_signal(int component, key_type key, string label, BaseMatrix* data){
    send_int(add_signal_flag, component, setup_tag, comm);

    send_key(key, component, setup_tag, comm);
    send_string(label, component, setup_tag, comm);
    send_matrix(data, component, setup_tag, comm);
}

void MpiInterface::add_op(int component, string op_string){
    send_int(add_op_flag, component, setup_tag, comm);

    send_string(op_string, component, setup_tag, comm);
}

void MpiInterface::add_probe(int component, key_type probe_key, string signal_string, float period){
    send_int(add_probe_flag, component, setup_tag, comm);

    send_key(probe_key, component, setup_tag, comm);
    send_string(signal_string, component, setup_tag, comm);
    send_int(period, component, setup_tag, comm);
}

void MpiInterface::finalize(){
    vector<MPISend*>::iterator send_it = master_chunk->mpi_sends.begin();
    for(; send_it != master_chunk->mpi_sends.end(); ++send_it){
        (*send_it)->set_communicator(comm);
    }

    vector<MPIRecv*>::iterator recv_it = master_chunk->mpi_recvs.begin();
    for(; recv_it != master_chunk->mpi_recvs.end(); ++recv_it){
        (*recv_it)->set_communicator(comm);
    }

    MPIBarrier* mpi_barrier = new MPIBarrier(comm);
    master_chunk->add_op(mpi_barrier);

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

void MpiInterface::gather_probe_data(map<key_type, vector<BaseMatrix*> >& probe_data,
                                     map<int, int>& probe_counts){
    key_type probe_key;
    map<int, int>::iterator count_it;
    int chunk_index, probe_count;

    cout << "Master gathering probe data from children..." << endl;

    vector<BaseMatrix*> new_data, data;

    for(count_it = probe_counts.begin(); count_it != probe_counts.end(); ++count_it){
        chunk_index = count_it->first;
        probe_count = count_it->second;

        if(chunk_index > 0){
            for(unsigned i = 0; i < probe_count; i++){
                probe_key = recv_key(chunk_index, probe_tag, comm);

                run_dbg("Master receiving probe from chunk " << chunk_index << endl
                        <<" with key " << probe_key << "..." << endl);

                int data_length = recv_int(chunk_index, probe_tag, comm);

                data = probe_data.at(probe_key);
                data.reserve(data.size() + new_data.size());

                for(int j = 0; j < data_length; j++){
                    BaseMatrix* matrix = recv_matrix(chunk_index, probe_tag, comm);
                    data.push_back(matrix);
                }

                probe_data[probe_key] = data;
            }
        }
    }

    cout << "Master done gathering probe data from children." << endl;

    MPI_Barrier(comm);
}

void MpiInterface::finish_simulation(){
    MPI_Finalize();
}