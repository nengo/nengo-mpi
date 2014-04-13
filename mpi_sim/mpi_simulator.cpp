#include "mpi_simulator.hpp"

void send_chunks(int num_chunks, MpiSimulatorChunk* chunks){
    MPI_Comm everyone;

    int argc = 0;
    char** argv;

    MPI_Init(&argc, &argv); 

    MPI_Comm_spawn("sim_mpi_worker", MPI_ARGV_NULL, num_chunks,
             MPI_INFO_NULL, 0, MPI_COMM_SELF, &everyone,  
             MPI_ERRCODES_IGNORE);

	mpi::intercommunicator intercomm(everyone, mpi::comm_duplicate); 
	mpi::communicator comm = intercomm.merge(false); 

#ifdef _DEBUG
	cout << "Master Rank in merged : " << comm.rank() << endl;	

    int buflen = 512;
    char name[buflen];
	MPI_Get_processor_name(name, &buflen);
    cout << "Master HOST: " << name << endl;
#endif

    int i;
    for(i = 0; i < num_chunks; ++i){
        comm.send(i+1, 1, chunks[i]);
    }

    //MPI_Finalize(); 
}