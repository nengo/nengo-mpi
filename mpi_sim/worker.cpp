#include <iostream>
#include <string>
#include <mpi.h>
#include <boost/mpi.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/intercommunicator.hpp>
#include <boost/serialization/string.hpp>

#include "simulator.hpp"

namespace mpi = boost::mpi;
using namespace std;

int main(int argc, char *argv[]) {

    int parent_size, parent_id, my_id, numprocs;

    // parent intercomm
    MPI_Comm parent;
    MPI_Init(&argc, &argv);

    MPI_Comm_get_parent(&parent);

    mpi::intercommunicator intercomm(parent, mpi::comm_duplicate);
    mpi::communicator comm = intercomm.merge(true);

    if (parent == MPI_COMM_NULL) {
        cout << "No parent!" << endl;
    }

    MPI_Comm_remote_size(parent, &parent_size);
    MPI_Comm_rank(parent, &parent_id) ;
    if (parent_size != 1) {
        cout << "Something's wrong with the parent" << endl;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &my_id) ;
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs) ;

    int buflen = 512;
    char name[buflen];
    MPI_Get_processor_name(name, &buflen);

#ifdef DEBUG
    cout << "I'm child process rank "<< my_id << " and we are " << numprocs << endl;
    cout << "The parent process has rank "<< parent_id << " and has size " << parent_size << endl;

    cout << "Child " << my_id << " host: " << name << endl;
    cout << "Child " << my_id << " rank in merged communicator: " << comm.rank() << endl;
#endif

    MpiSimulatorChunk chunk;

    // Recv the chunk from master
    comm.recv(0, 1, chunk);

    // Send a validation string to master to confirm that
    // the chunk was transferred properly
    string validation_string = chunk.to_string();
    comm.send(0, 2, validation_string);

    chunk.fix_mpi_waits();

    map<int, MPISend*>::iterator send_it;
    for(send_it = chunk.mpi_sends.begin(); send_it != chunk.mpi_sends.end(); ++send_it){
        send_it->second->comm = &comm;
    }

    map<int, MPIRecv*>::iterator recv_it;
    for(recv_it = chunk.mpi_recvs.begin(); recv_it != chunk.mpi_recvs.end(); ++recv_it){
        recv_it->second->comm = &comm;
    }

    // Wait for the signal to run the simulation
    int steps;
    broadcast(comm, steps, 0);
    cout << "Child " << my_id << " got the signal to start simulation!: " << steps << " steps." << endl;

    chunk.run_n_steps(steps);
    comm.barrier();

    MPI_Finalize();
    return 0;
}