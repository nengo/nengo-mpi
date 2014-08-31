#include "mpi_simulator.hpp"

void send_chunks(list<MpiSimulatorChunk*> chunks){
    cout << "C++: SENDING CHUNKS";
    int num_chunks = chunks.size();
    MPI_Comm everyone;

    int argc = 0;
    char** argv;

    cout << "C++: Before";
    MPI_Init(&argc, &argv);
    cout << "C++: After";

    MPI_Comm_spawn("sim_mpi_worker", MPI_ARGV_NULL, num_chunks,
             MPI_INFO_NULL, 0, MPI_COMM_SELF, &everyone,
             MPI_ERRCODES_IGNORE);
    cout << "C++: After1";

    mpi::intercommunicator intercomm(everyone, mpi::comm_duplicate);
    mpi::communicator comm = intercomm.merge(false);

#ifdef _DEBUG
    cout << "Master Rank in merged : " << comm.rank() << endl;

    int buflen = 512;
    char name[buflen];
    MPI_Get_processor_name(name, &buflen);
    cout << "Master HOST: " << name << endl;
#endif
    cout << "C++: After2";

    int i;
    string original_string, remote_string;
    list<MpiSimulatorChunk*>::const_iterator it;

    for(it = chunks.begin(); it != chunks.end(); ++it){
        cout << "C++: After3";

        // Send the chunk
        comm.send(i+1, 1, **it);

        // Make sure the chunk was sent correctly
        comm.recv(i+1, 2, remote_string);
        original_string = (**it).to_string();
        assert(original_string == remote_string);

        //TODO: Free the chunks on this node!
    }

    //MPI_Finalize();
}