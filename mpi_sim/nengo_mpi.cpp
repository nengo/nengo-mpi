#include <iostream>
#include <string>
#include <memory>

#include <mpi.h>
#include <hdf5.h>

#include "optionparser.h"

#include "chunk.hpp"
#include "mpi_simulator.hpp"

using namespace std;

// comm: The communicator for the worker to communicate on. Must
// be an intracommunicator involving all processes, with the master
// process having rank 0.
void start_worker(MPI_Comm comm){

    int my_id, n_processors;
    MPI_Comm_rank(comm, &my_id);
    MPI_Comm_size(comm, &n_processors);

    int buflen = 512;
    char name[buflen];
    MPI_Get_processor_name(name, &buflen);

    dbg("Hello world! I'm a nengo_mpi worker process with "
        "rank "<< my_id << " on host " << name << "." << endl);

    MPI_Status status;

    dbg("Reading merged...");
    int mpi_merged = bcast_recv_int(comm);

    dbg("Reading filename...");
    string filename = recv_string(0, setup_tag, comm);

    dbg("Creating chunk...");
    MpiSimulatorChunk chunk(my_id, n_processors, bool(mpi_merged));

    // Use parallel property lists
    hid_t file_plist = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(file_plist, comm, MPI_INFO_NULL);

    hid_t read_plist = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(read_plist, H5FD_MPIO_INDEPENDENT);

    dbg("Loading from file...");
    chunk.from_file(filename, file_plist, read_plist);
    chunk.finalize_build(comm);

    H5Pclose(file_plist);
    H5Pclose(read_plist);

    // Worker barrier 1
    MPI_Barrier(comm);

    dbg("Worker " << my_id << " waiting for signal to start simulation...");
    int steps;
    MPI_Bcast(&steps, 1, MPI_INT, 0, comm);

    dbg("Worker " << my_id << " got the signal to start simulation: "
        << steps << " steps." << endl);

    chunk.run_n_steps(steps, false);

    // Worker barrier 2
    MPI_Barrier(comm);

    if(!chunk.is_logging()){
        // If we're not logging, send the probe data back to the master
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

    // Worker barrier 3
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

enum serialOptionIndex {UNKNOWN, HELP, NO_PROG, LOG, MERGED};

const option::Descriptor serial_usage[] =
{
 {UNKNOWN, 0, "" , "",          option::Arg::None, "USAGE: nengo_mpi [options] <network file> <simulation time> \n\n"
                                                   "where <network file> is a file specifying a network to simulate,\n"
                                                   "and <simulation time> is the amount of time to simulate the network\n"
                                                   "for, in seconds. Simulation results are logged to an HDF5 file.\n"
                                                   "Note that the options must come before the network file.\n"
                                                   "Options:" },
 {HELP,     0, "" , "help",     option::Arg::None, "  --help  \tPrint usage and exit." },
 {NO_PROG,  0, "",  "noprog",   option::Arg::None, "  --noprog  \tSupply to omit the progress bar." },
 {LOG,      0, "",  "log",      Arg::NonEmpty,     "  --log  \tName of file to log results to using HDF5. "
                                                               "If not specified, the log filename is the same as the "
                                                               "name of the network file, but with the .h5 extension."},
 {MERGED,   0, "",  "merged",   option::Arg::None, "  --merged  \tSupply to use merged communication mode."},
 {UNKNOWN,  0, "" , ""   ,      option::Arg::None, "\nExamples:\n"
                                                   "  nengo_mpi --noprog basal_ganglia.net 1.0\n"
                                                   "  nengo_mpi --log ~/spaun_results.h5 spaun.net 7.5\n" },
 {0,0,0,0,0,0}
};

int start_master(int argc, char **argv){

    argc -= (argc > 0); argv += (argc > 0); // skip program name argv[0] if present
    option::Stats  stats(serial_usage, argc, argv);
    option::Option options[stats.options_max], buffer[stats.buffer_max];
    option::Parser parse(serial_usage, argc, argv, options, buffer);

    if (parse.error()){
        return 0;
    }

    if (options[HELP] || argc == 0) {
        option::printUsage(std::cout, serial_usage);
        return 0;
    }

    string net_filename;
    float sim_length;

    // Handle mandatory options.
    if(parse.nonOptionsCount() != 2){
        cout << "Please specify a network to simulate and a simulation time." << endl << endl;
        option::printUsage(std::cout, serial_usage);
        return 0;
    }else{
        net_filename = parse.nonOptions()[0];

        string s_sim_length(parse.nonOptions()[1]);

        try{
            sim_length = boost::lexical_cast<float>(s_sim_length);
        }catch(const boost::bad_lexical_cast& e){
            stringstream msg;
            msg << "Specified simulation time, " << s_sim_length
                << ", could not be interpreted as a float." << endl;
            throw runtime_error(msg.str());
        }
    }

    cout << "Will load network from file: " << net_filename << "." << endl;
    cout << "Will run simulation for " << sim_length << " second(s)." << endl;

    bool show_progress = !bool(options[NO_PROG]);
    cout << "Show progress bar: " << show_progress << endl;

    bool mpi_merged = bool(options[MERGED]);
    cout << "Merged communication mode: " << mpi_merged << endl;

    string log_filename;
    if(options[LOG]){
        log_filename = options[LOG].arg;
    }else{
        log_filename = net_filename.substr(0, net_filename.find_last_of(".")) + ".h5";
    }
    cout << "Will write simulation results to " << log_filename << endl;

    cout << "Building network..." << endl;
    auto sim = unique_ptr<MpiSimulator>(new MpiSimulator(mpi_merged));
    sim->from_file(net_filename);
    sim->finalize_build();

    cout << "Done building network..." << endl;

    cout << "dt: " << sim->dt << endl;
    int num_steps = int(round(sim_length / sim->dt));

    cout << "Num steps: " << num_steps << endl;
    cout << "Running simulation..." << endl;

    sim->run_n_steps(num_steps, show_progress, log_filename);

    return 0;
}

int main(int argc, char **argv){

    MPI_Init(&argc, &argv);

    MPI_Comm parent;
    MPI_Comm_get_parent(&parent);

    if (parent != MPI_COMM_NULL){
        // We have a parent, so we must be a worker process spawned
        // by the master, which must be running through python.
        // Get a communicator that includes the parent and all children.

        MPI_Comm everyone;
        MPI_Intercomm_merge(parent, true, &everyone);
        start_worker(everyone);
    }else{
        // We don't have a parent, so we must be either the master process
        // or a worker process, in either case created at the beginning of
        // a call to nengo_mpi run using mpirun. Can use the MPI_COMM_WORLD
        // communicator, as it will contain all processes in the simulation.

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