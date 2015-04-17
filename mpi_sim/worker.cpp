#include <iostream>
#include <string>
#include <memory>

#include <mpi.h>

#include "optionparser.h"

#include "chunk.hpp"
#include "simulator.hpp"
#include "utils.hpp"

using namespace std;

// This file can be used in two ways. First, if using nengo_mpi from python, this
// code is the entry point for the workers that are spawned by the initial process.
// If using nengo_mpi straight from C++, then this file is the entry point for ALL
// processes in the simulation. In that case, the process with rank 0 will create an
// MpiSimulator object and load a built nengo network from a file specified on the
// command line, and the rest of the processes will jump straight into start_worker.

// comm: The communicator for the worker to communicate on. Must
// be an intracommunicator involving all processes, with the master
// process having rank 0.
void start_worker(MPI_Comm comm){

    int my_id, num_procs;
    MPI_Comm_rank(comm, &my_id);
    MPI_Comm_size(comm, &num_procs);

    int buflen = 512;
    char name[buflen];
    MPI_Get_processor_name(name, &buflen);

    dbg("Hello world! I'm a nengo_mpi worker process with "
        "rank "<< my_id << " on host " << name << "." << endl);

    MPI_Status status;

    string chunk_label = recv_string(0, setup_tag, comm);

    dtype dt = recv_dtype(0, setup_tag, comm);

    MpiSimulatorChunk chunk(my_id, chunk_label, dt);

    int s = 0;
    string op_string;

    key_type probe_key;
    string signal_string;
    dtype period;

    dbg("Worker " << my_id  << " receiving network...");

    while(1){
        s = recv_int(0, setup_tag, comm);

        if(s == add_signal_flag){
            dbg("Worker " << my_id  << " receiving signal.");

            key_type key = recv_key(0, setup_tag, comm);

            string label = recv_string(0, setup_tag, comm);

            unique_ptr<BaseSignal> data = recv_matrix(0, setup_tag, comm);

            dbg("Worker " << my_id  << " done receiving signal.");

            dbg("key; " << key);
            dbg("label; " << key);
            dbg("data; " << *data);

            chunk.add_base_signal(key, label, move(data));

        }else if(s == add_op_flag){
            dbg("Worker " << my_id  << " receiving operator.");

            string op_string = recv_string(0, setup_tag, comm);

            dbg("Worker " << my_id  << " done receiving operator.");

            chunk.add_op(op_string);

        }else if(s == add_probe_flag){
            dbg("Worker " << my_id  << " receiving probe.");

            key_type probe_key = recv_key(0, setup_tag, comm);

            string signal_string = recv_string(0, setup_tag, comm);

            dtype period = recv_dtype(0, setup_tag, comm);

            dbg("Worker " << my_id  << " done receiving probe.");

            chunk.add_probe(probe_key, signal_string, period);

        }else if(s == stop_flag){
            dbg("Worker " << my_id  << " done receiving network.");
            break;

        }else{
            throw runtime_error("Worker received invalid flag from master.");
        }
    }

    dbg("Worker " << my_id << " setting up simulation log...");
    auto sim_log = unique_ptr<SimulationLog>(
        new ParallelSimulationLog(num_procs, my_id, dt, comm));

    chunk.set_simulation_log(move(sim_log));

    dbg("Worker " << my_id << " setting up MPI operators...");
    chunk.set_communicator(comm);
    chunk.add_op(unique_ptr<Operator>(new MPIBarrier(comm)));

    dbg("Worker " << my_id << " waiting for signal to start simulation...");

    int steps;
    MPI_Bcast(&steps, 1, MPI_INT, 0, comm);

    dbg("Worker " << my_id << " got the signal to start simulation: "
        << steps << " steps." << endl);

    chunk.run_n_steps(steps, false);

    MPI_Barrier(comm);

    if(!chunk.is_logging()){
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
    }

    MPI_Barrier(comm);

    chunk.close_simulation_log();

    MPI_Finalize();
}

struct Arg: public option::Arg
 {
     static void printError(const char* msg1, const option::Option& opt, const char* msg2)
     {
         fprintf(stderr, "ERROR: %s", msg1);
         fwrite(opt.name, opt.namelen, 1, stderr);
         fprintf(stderr, "%s", msg2);
     }

     static option::ArgStatus NonEmpty(const option::Option& option, bool msg)
     {
         if (option.arg != 0 && option.arg[0] != 0)
             return option::ARG_OK;

         if (msg) printError("Option '", option, "' requires a non-empty argument\n");
         return option::ARG_ILLEGAL;
     }

     static option::ArgStatus Numeric(const option::Option& option, bool msg)
     {
         char* endptr = 0;
         if (option.arg != 0 && strtol(option.arg, &endptr, 10)){};
         if (endptr != option.arg && *endptr == 0)
             return option::ARG_OK;

         if (msg) printError("Option '", option, "' requires a numeric argument\n");
         return option::ARG_ILLEGAL;
     }
 };

enum  optionIndex { UNKNOWN, HELP, PROC, LOG, NET, PROGRESS, TIME};

const option::Descriptor usage[] =
{
 {UNKNOWN, 0, "" , "",          option::Arg::None, "USAGE: example [options]\n\n"
                                                   "Options:" },
 {HELP,     0, "" , "help",     option::Arg::None, "  --help  \tPrint usage and exit." },
 {NET,      0, "",  "net",      Arg::NonEmpty,     "  --net,  \tName of network to simulate. Mandatory" },
 {TIME,     0, "t", "time",     Arg::NonEmpty, "  --time, -t  \tTime to simulate for, in seconds. Mandatory" },
 {PROC,     0, "p", "proc",     Arg::NonEmpty, "  --proc, -p  \tNumber of processors to use. "
                                                                   "If not specified, will be read from the network file."},
 {LOG,      0, "",  "log",      Arg::NonEmpty, "  --log,  \tName of file to log results to. "
                                                               "If not specified, results printed at end of simulation." },
 {PROGRESS, 0, "",  "progress", option::Arg::None, "  --progress, \tSupply to show progress bar." },
 {UNKNOWN,  0, "" , ""   ,      option::Arg::None, "\nExamples:\n"
                                                   "  mpi_sim_worker --net basal_ganglia.net -t 1.0\n"
                                                   "  mpi_sim_worker --net spaun.net -t 7.5 -p 1024\n" },
 {0,0,0,0,0,0}
};

void start_master(int argc, char **argv){

    argc -= (argc > 0); argv += (argc > 0); // skip program name argv[0] if present
    option::Stats  stats(usage, argc, argv);
    option::Option options[stats.options_max], buffer[stats.buffer_max];
    option::Parser parse(usage, argc, argv, options, buffer);

    if (parse.error()){
        return;
    }

    if (options[HELP] || argc == 0) {
        option::printUsage(std::cout, usage);
        return;
    }

    if(!options[NET]){
        cout << "Please specify network to simulate." << endl;
        return;
    }

    string net_filename(options[NET].arg);
    cout << "Loading network from file: " << net_filename << "." << endl;

    if(!options[TIME]){
        cout << "Please specify a simulation length" << endl;
        return;
    }

    string s_sim_length(options[TIME].arg);

    float sim_length;
    try{
        sim_length = boost::lexical_cast<float>(s_sim_length);
    }catch(const boost::bad_lexical_cast& e){
        stringstream msg;
        msg << "Specified simulation time could not be interpreted as a float." << endl;
        throw runtime_error(msg.str());
    }

    cout << "Will run simulation for " << sim_length << " second(s)." << endl;

    bool show_progress = bool(options[PROGRESS]);

    string log_filename = options[LOG] ? options[LOG].arg : "";

    int n_processors;
    if(options[PROC]){
        string s_n_processors(options[PROC].arg);
        try{
            n_processors = boost::lexical_cast<int>(s_n_processors);
        }catch(const boost::bad_lexical_cast& e){
            stringstream msg;
            msg << "Specified a value for n_processors that could not be interpreted as an int." << endl;
            throw runtime_error(msg.str());
        }
    }

    cout << "Building network..." << endl;
    unique_ptr<Simulator> sim = create_simulator_from_file(net_filename);
    cout << "Done building network..." << endl;

    cout << "dt: " << sim->dt << endl;
    int num_steps = int(sim_length / sim->dt);

    cout << "Num steps: " << num_steps << endl;
    cout << "Running simulation..." << endl;
    if(log_filename.size() != 0){
        cout << "Logging simulation results to " << log_filename << endl;
    }

    sim->run_n_steps(num_steps, show_progress, log_filename);

    if(log_filename.size() == 0){
        for(auto& key : sim->get_probe_keys()){
            vector<unique_ptr<BaseSignal>> probe_data = sim->get_probe_data(key);

            cout << "Probe data for key: " << key << endl;

            for(auto& pd : probe_data){
                cout << *pd << endl;
            }
        }
    }
}

int main(int argc, char **argv){

    MPI_Init(&argc, &argv);

    MPI_Comm parent;
    MPI_Comm_get_parent(&parent);

    if (parent != MPI_COMM_NULL){
        MPI_Comm everyone;

        // We have a parent, so get a communicator that includes the
        // parent and all children
        MPI_Intercomm_merge(parent, true, &everyone);
        start_worker(everyone);
    }else{
        int mpi_size;
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

        if(mpi_size == 1){
            // No parent, and only one process. Corresponds to a serial
            // simulation using C++ directly (skipping python).
            MPI_Finalize();

            start_master(argc, argv);
            return 0;
        }

        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        if(rank == 0){
            start_master(argc, argv);
        }else{
            start_worker(MPI_COMM_WORLD);
        }
    }

    return 0;
}