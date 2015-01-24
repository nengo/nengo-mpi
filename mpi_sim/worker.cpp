#include <iostream>
#include <string>

#include <mpi.h>

#include <boost/mpi.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/intercommunicator.hpp>

#include "flags.hpp"
#include "simulator.hpp"

namespace mpi = boost::mpi;
using namespace std;

// This file can be used in two ways. First, if using nengo_mpi from python, this
// code is the entry point for the workers that are spawned by the initial process.
// If using nengo_mpi straight from C++, then this file is the entry point for all
// processes in the simulation. In that case, the process with rank 0 will create an
// MpiSimulator object and load a built nengo network from a file specified on the
// command line, and the rest of the processes will jump straight into start_worker.

// comm: The communicator for the worker to communicate on.
void start_worker(mpi::communicator comm){

    int my_id = comm.rank();
    int num_procs = comm.size();

    int buflen = 512;
    char name[buflen];
    MPI_Get_processor_name(name, &buflen);

    cout << "Hello world! I'm a nengo_mpi worker process with rank "<< my_id << " on host " << name << "." << endl;

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
            dbg("Worker " << my_id  << " receiving signal.");

            comm.recv(0, tag, key);
            comm.recv(0, tag, label);
            comm.recv(0, tag, data);

            chunk.add_base_signal(key, label, data);

        }else if(s == add_op_flag){
            dbg("Worker " << my_id  << " receiving operator.");
            comm.recv(0, tag, op_string);

            chunk.add_op(op_string);

        }else if(s == add_probe_flag){
            dbg("Worker " << my_id  << " receiving probe.");
            comm.recv(0, tag, probe_key);
            comm.recv(0, tag, signal_string);
            comm.recv(0, tag, period);

            chunk.add_probe(probe_key, signal_string, period);

        }else if(s == stop_flag){
            dbg("Worker " << my_id  << " done building.");
            break;

        }else{
            throw runtime_error("Worker received invalid flag from master.");
        }
    }

    dbg("Worker setting up MPI operators..");
    chunk.setup_mpi_waits();

    map<int, MPISend*>::iterator send_it;
    for(send_it = chunk.mpi_sends.begin(); send_it != chunk.mpi_sends.end(); ++send_it){
        send_it->second->comm = &comm;
    }

    map<int, MPIRecv*>::iterator recv_it;
    for(recv_it = chunk.mpi_recvs.begin(); recv_it != chunk.mpi_recvs.end(); ++recv_it){
        recv_it->second->comm = &comm;
    }

    MPIBarrier* mpi_barrier = new MPIBarrier(&comm);
    chunk.add_op(mpi_barrier);

    dbg("Worker waiting for signal to start simulation.");

    int steps;
    broadcast(comm, steps, 0);
    cout << "Worker process " << my_id << " got the signal to start simulation: " << steps << " steps." << endl;

    chunk.run_n_steps(steps, false);
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
}

int main(int argc, char **argv){

    MPI_Init(&argc, &argv);

    MPI_Comm parent;
    MPI_Comm_get_parent(&parent);

    if (parent != MPI_COMM_NULL){
        mpi::intercommunicator intercomm(parent, mpi::comm_duplicate);
        mpi::communicator comm = intercomm.merge(true);

        start_worker(comm);
    }else{
        mpi::communicator comm;
        int rank = comm.rank();

        if(rank == 0){
            if(argc < 1){
                cout << "Please specify a file to load" << endl;
                return 0;
            }

            if(argc < 2){
                cout << "Please specify a simulation length" << endl;
                return 0;
            }

            string filename = argv[1];
            bool spawn = false;

            MpiSimulator mpi_sim(filename, spawn);

            int num_steps = boost::lexical_cast<int>(argv[2]);
            mpi_sim.run_n_steps(num_steps, true);
        }
        else{
            start_worker(comm);
        }
    }

    return 0;
}