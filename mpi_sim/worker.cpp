#include <iostream>
#include <string>
#include <memory>

#include <mpi.h>

#include "chunk.hpp"
#include "simulator.hpp"
#include "mpi_interface.hpp"

using namespace std;

// This file can be used in two ways. First, if using nengo_mpi from python, this
// code is the entry point for the workers that are spawned by the initial process.
// If using nengo_mpi straight from C++, then this file is the entry point for ALL
// processes in the simulation. In that case, the process with rank 0 will create an
// MpiSimulator object and load a built nengo network from a file specified on the
// command line, and the rest of the processes will jump straight into start_worker.

// comm: The communicator for the worker to communicate on.
void start_worker(MPI_Comm comm){

    int my_id, num_procs;
    MPI_Comm_rank(comm, &my_id);
    MPI_Comm_size(comm, &num_procs);

    int buflen = 512;
    char name[buflen];
    MPI_Get_processor_name(name, &buflen);

    cout << "Hello world! I'm a nengo_mpi worker process with "
            "rank "<< my_id << " on host " << name << "." << endl;

    MPI_Status status;

    string chunk_label = recv_string(0, setup_tag, comm);

    dtype dt = recv_dtype(0, setup_tag, comm);

    MpiSimulatorChunk chunk(chunk_label, dt);

    int s = 0;
    string op_string;

    key_type probe_key;
    string signal_string;
    dtype period;

    while(1){
        s = recv_int(0, setup_tag, comm);

        if(s == add_signal_flag){
            dbg("Worker " << my_id  << " receiving signal.");

            key_type key = recv_key(0, setup_tag, comm);

            string label = recv_string(0, setup_tag, comm);

            unique_ptr<BaseSignal> data = recv_matrix(0, setup_tag, comm);

            chunk.add_base_signal(key, label, move(data));

            dbg("Worker " << my_id  << " done receiving signal.");
            dbg("key; " << key);
            dbg("label; " << key);
            dbg("data; " << *data);

        }else if(s == add_op_flag){
            dbg("Worker " << my_id  << " receiving operator.");

            string op_string = recv_string(0, setup_tag, comm);

            chunk.add_op(op_string);

        }else if(s == add_probe_flag){
            dbg("Worker " << my_id  << " receiving probe.");

            key_type probe_key = recv_key(0, setup_tag, comm);

            string signal_string = recv_string(0, setup_tag, comm);

            dtype period = recv_dtype(0, setup_tag, comm);

            chunk.add_probe(probe_key, signal_string, period);

        }else if(s == stop_flag){
            dbg("Worker " << my_id  << " done building.");
            break;

        }else{
            throw runtime_error("Worker received invalid flag from master.");
        }
    }

    dbg("Worker setting up MPI operators..");


    chunk.set_communicator(comm);
    chunk.add_op(unique_ptr<Operator>(new MPIBarrier(comm)));

    dbg("Worker waiting for signal to start simulation.");

    int steps;
    MPI_Bcast(&steps, 1, MPI_INT, 0, comm);

    cout << "Worker process " << my_id
         << " got the signal to start simulation: " << steps << " steps." << endl;

    chunk.run_n_steps(steps, false);

    MPI_Barrier(comm);

    for(auto& pair : chunk.probe_map){
        key_type key = pair.first;
        shared_ptr<Probe>& probe = pair.second;

        send_key(key, 0, probe_tag, comm);

        vector<unique_ptr<BaseSignal>> probe_data = probe->harvest_data();

        send_int(probe_data.size(), 0, probe_tag, comm);

        for(auto& pd : probe_data){
            send_matrix(move(pd), 0, probe_tag, comm);
        }
    }

    MPI_Barrier(comm);

    MPI_Finalize();
}

int main(int argc, char **argv){

    MPI_Init(&argc, &argv);

    MPI_Comm parent;
    MPI_Comm_get_parent(&parent);

    if (parent != MPI_COMM_NULL){
        MPI_Comm everyone;
        MPI_Intercomm_merge(parent, true, &everyone);
        start_worker(everyone);
    }else{
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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

            for(auto& key : mpi_sim.get_probe_keys()){
                vector<unique_ptr<BaseSignal>> probe_data = mpi_sim.get_probe_data(key);

                cout << "Probe data for key: " << key << endl;

                for(auto& pd : probe_data){
                    cout << *pd << endl;
                }
            }
        }
        else{
            start_worker(MPI_COMM_WORLD);
        }
    }

    return 0;
}