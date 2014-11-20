#include "mpi_simulator.hpp"

void MpiInterface::initialize_chunks(MpiSimulatorChunk* chunk, list<MpiSimulatorChunk*> remote_chunks){

    master_chunk = chunk;

    cout << "C++: initializing master chunk\n";
    master_chunk->fix_mpi_waits();

    map<int, MPISend*>::iterator send_it;
    for(send_it = master_chunk->mpi_sends.begin(); send_it != master_chunk->mpi_sends.end(); ++send_it){
        send_it->second->comm = &comm;
    }

    map<int, MPIRecv*>::iterator recv_it;
    for(recv_it = master_chunk->mpi_recvs.begin(); recv_it != master_chunk->mpi_recvs.end(); ++recv_it){
        recv_it->second->comm = &comm;
    }

    cout << "C++: sending remote chunks\n";
    int num_remote_chunks = remote_chunks.size();
    MPI_Comm everyone;

    int argc = 0;
    char** argv;

    cout << "Master initing MPI..." << endl;
    MPI_Init(&argc, &argv);
    cout << "Master finished initing MPI." << endl;

    cout << "Master spawning " << num_remote_chunks << " children..." << endl;
    MPI_Comm_spawn("mpi_sim_worker", MPI_ARGV_NULL, num_remote_chunks,
             MPI_INFO_NULL, 0, MPI_COMM_SELF, &everyone,
             MPI_ERRCODES_IGNORE);
    cout << "Master finished spawning children." << endl;

    mpi::intercommunicator intercomm(everyone, mpi::comm_duplicate);
    comm = intercomm.merge(false);

#ifdef DEBUG
    int buflen = 512;
    char name[buflen];
    MPI_Get_processor_name(name, &buflen);
    cout << "Master host: " << name << endl;
    cout << "Master rank in merged communicator: " << comm.rank() << " (should be 0)." << endl;
#endif

    int chunk_index = 1;
    string original_string, remote_string;
    list<MpiSimulatorChunk*>::const_iterator it;

    for(it = remote_chunks.begin(); it != remote_chunks.end(); ++it){

        cout << "Master sending chunk " << chunk_index << "..." << endl;

        // Send the chunk
        comm.send(chunk_index, 1, **it);

        cout << "Master finished sending chunk " << chunk_index << endl;

        cout << "Master receiving chunk " << chunk_index << " for validation..." << endl;

        // Make sure the chunk was sent correctly
        comm.recv(chunk_index, 2, remote_string);

        cout << "Master finished receiving chunk " << chunk_index << " for validation." << endl;

        dbg("Remote string: " <<  chunk_index << endl);
        dbg(remote_string << endl);

        original_string = (**it).to_string();
        assert(original_string == remote_string);

        cout << "Chunk " << chunk_index << " sent successfully." << endl;

        chunk_index++;

        //TODO: Free the remote_chunks on this node!
    }
}

void MpiInterface::run_n_steps(int steps){
    cout << "Simulating..." << endl;
    broadcast(comm, steps, 0);

    dbg("Master mpi ops: " << endl << master_chunk->print_maps());

    cout << "Master starting simulation!: " << steps << " steps." << endl;

    master_chunk->run_n_steps(steps);

    comm.barrier();

    cout << "Finished simulation." << endl;
}

void MpiInterface::gather_probe_data(map<key_type, vector<Matrix*>*>& probe_data,
                                     map<int, int>& probe_counts){
    key_type probe_key;
    vector<Matrix*>* data = NULL;
    map<int, int>::iterator count_it;
    int chunk_index, probe_count;

    cout << "Master gathering probe data from children..." << endl;

    for(count_it = probe_counts.begin(); count_it != probe_counts.end(); ++count_it){
        chunk_index = count_it->first;
        probe_count = count_it->second;

        if(chunk_index > 0){
            for(unsigned i = 0; i < probe_count; i++){
                data = new vector<Matrix*>();

                cout << "Master receiving probe from chunk " << chunk_index;
                comm.recv(chunk_index, 3, probe_key);
                cout << " with key " << probe_key << "..." << endl;
                comm.recv(chunk_index, 3, *data);
                cout << "Done receiving probe data." << endl;

                probe_data[probe_key] = data;
            }
        }
    }

    cout << "Master done gathering probe data from children." << endl;

    comm.barrier();
}

void MpiInterface::finish_simulation(){
    MPI_Finalize();
}