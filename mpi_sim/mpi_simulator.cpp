#include "mpi_simulator.hpp"

void MpiInterface::send_chunks(list<MpiSimulatorChunk*> chunks){
    cout << "C++: SENDING CHUNKS\n";
    int num_chunks = chunks.size();
    MPI_Comm everyone;

    int argc = 0;
    char** argv;

    cout << "Master initing MPI..." << endl;
    MPI_Init(&argc, &argv);
    cout << "Master finished initing MPI." << endl;

    cout << "Master spawning " << num_chunks << " children..." << endl;
    MPI_Comm_spawn("mpi_sim_worker", MPI_ARGV_NULL, num_chunks,
             MPI_INFO_NULL, 0, MPI_COMM_SELF, &everyone,
             MPI_ERRCODES_IGNORE);
    cout << "Master finished spawning children." << endl;

    mpi::intercommunicator intercomm(everyone, mpi::comm_duplicate);
    comm = intercomm.merge(false);
    //mpi::communicator comm = intercomm.merge(false);

#ifdef DEBUG
    int buflen = 512;
    char name[buflen];
    MPI_Get_processor_name(name, &buflen);
    cout << "Master host: " << name << endl;
    cout << "Master rank in merged communicator: " << comm.rank() << " (should be 0)." << endl;
#endif

    int i = 0;
    string original_string, remote_string;
    list<MpiSimulatorChunk*>::const_iterator it;

    for(it = chunks.begin(); it != chunks.end(); ++it){

        cout << "Master sending chunk " << i << "..." << endl;
        // Send the chunk
        comm.send(i+1, 1, **it);

        cout << "Master finished sending chunk " << i << endl;

        cout << "Master receiving chunk " << i << "..." << endl;

        // Make sure the chunk was sent correctly
        comm.recv(i+1, 2, remote_string);

        cout << "Master finished receiving chunk " << i << endl;

        cout << "Remote string, i: " << i << endl;
        cout << remote_string << endl;

        original_string = (**it).to_string();
        assert(original_string == remote_string);

        i++;

        //TODO: Free the chunks on this node!
    }

    //MPI_Finalize();
}

void MpiInterface::start_simulation(int steps){
    broadcast(comm, steps, 0);
    comm.barrier();
    MPI_Finalize();
}