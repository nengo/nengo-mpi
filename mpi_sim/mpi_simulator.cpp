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

    int i = 0, chunk_index;
    string original_string, remote_string;
    list<MpiSimulatorChunk*>::const_iterator it;

    for(it = remote_chunks.begin(); it != remote_chunks.end(); ++it){

        chunk_index = i + 1;

        cout << "Master sending chunk " << chunk_index << "..." << endl;

        // Send the chunk
        comm.send(chunk_index, 1, **it);

        cout << "Master finished sending chunk " << chunk_index << endl;

        cout << "Master receiving chunk " << chunk_index << " for validation..." << endl;

        // Make sure the chunk was sent correctly
        comm.recv(chunk_index, 2, remote_string);

        cout << "Master finished receiving chunk " << chunk_index << " for validation." << endl;

        dbg("Remote string, i: " << i << endl);
        dbg(remote_string << endl);

        original_string = (**it).to_string();
        assert(original_string == remote_string);

        cout << "Chunk " << i << " sent successfully." << endl;

        i++;

        //TODO: Free the remote_chunks on this node!
    }
}

void MpiInterface::start_simulation(int steps){
    cout << "Simulating..." << endl;
    broadcast(comm, steps, 0);

    dbg("Master mpi ops: " << endl << master_chunk->print_maps());

    cout << "Master starting simulation!: " << steps << " steps." << endl;

    master_chunk->run_n_steps(steps);
    comm.barrier();

    cout << "Finished simulation." << endl;

    MPI_Finalize();
}