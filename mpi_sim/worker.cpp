#include <iostream>
#include <string>
#include <mpi.h>
#include <boost/mpi.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/intercommunicator.hpp>
#include <boost/serialization/string.hpp>

#include "flags.hpp"
#include "simulator.hpp"

namespace mpi = boost::mpi;
using namespace std;

int main(int argc, char *argv[]) {

    int parent_size, parent_id;

    // parent intercomm
    MPI_Comm parent;
    MPI_Init(&argc, &argv);

    MPI_Comm_get_parent(&parent);

    mpi::intercommunicator intercomm(parent, mpi::comm_duplicate);
    mpi::communicator comm = intercomm.merge(true);

    if (parent == MPI_COMM_NULL) {
        cout << "No parent." << endl;
    }

    MPI_Comm_remote_size(parent, &parent_size);
    MPI_Comm_rank(parent, &parent_id) ;
    if (parent_size != 1) {
        cout << "Something's wrong with the parent" << endl;
    }

    int my_id = comm.rank();
    int num_procs = comm.size();

    //MPI_Comm_rank(MPI_COMM_WORLD, &my_id) ;
    //MPI_Comm_size(MPI_COMM_WORLD, &num_procs) ;

    int buflen = 512;
    char name[buflen];
    MPI_Get_processor_name(name, &buflen);

    cout << "Hello world! I'm a nengo_mpi worker process with rank "<< my_id << " of " << num_procs << endl;

    float dt;
    string chunk_label;
    int tag = 1;

    comm.recv(0, tag, chunk_label);
    comm.recv(0, tag, dt);

    MpiSimulatorChunk chunk(chunk_label, dt);

    int s = 0;
    key_type key;
    string label;
    BaseMatrix data;
    string op_string;

    key_type probe_key;
    string signal_string;
    float period;

    while(1){
        comm.recv(0, tag, s);

        if(s == add_signal_flag){
            comm.recv(0, tag, key);
            comm.recv(0, tag, label);
            comm.recv(0, tag, data);

            chunk.add_base_signal(key, label, data);

        }else if(s == add_op_flag){
            comm.recv(0, tag, op_string);

            chunk.add_op(op_string);

        }else if(s == add_probe_flag){
            comm.recv(0, tag, probe_key);
            comm.recv(0, tag, signal_string);
            comm.recv(0, tag, period);

            chunk.add_probe(probe_key, signal_string, period);

        }else if(s == stop_flag){
            break;

        }else{
            throw runtime_error("Worker received invalid flag from master.");
        }
    }

    chunk.setup_mpi_waits();

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
    cout << "Worker process " << my_id << " got the signal to start simulation: " << steps << " steps." << endl;

    chunk.run_n_steps(steps);
    comm.barrier();

    map<key_type, Probe*>::iterator probe_it;
    vector<BaseMatrix*> probe_data;

    for(probe_it = chunk.probe_map.begin(); probe_it != chunk.probe_map.end(); ++probe_it){
        comm.send(0, 3, probe_it->first);

        probe_data = probe_it->second->get_data();
        comm.send(0, 3, probe_data);
        probe_it->second->clear(true);
    }

    comm.barrier();

    MPI_Finalize();
    return 0;
}