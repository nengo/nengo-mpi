#include <iostream>
#include <string>
#include <memory>

#include "optionparser.h"
#include "simulator.hpp"

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


enum serialOptionIndex {UNKNOWN, HELP, PROC, LOG, NET, PROGRESS, TIME};

const option::Descriptor serial_usage[] =
{
 {UNKNOWN, 0, "" , "",          option::Arg::None, "USAGE: nengo_cpp [options]\n\n"
                                                   "Options:" },
 {HELP,     0, "" , "help",     option::Arg::None, "  --help  \tPrint usage and exit." },
 {NET,      0, "",  "net",      Arg::NonEmpty,     "  --net,  \tName of network to simulate. Mandatory" },
 {TIME,     0, "t", "time",     Arg::NonEmpty,     "  --time, -t  \tTime to simulate for, in seconds. Mandatory" },
 {LOG,      0, "",  "log",      Arg::NonEmpty,     "  --log,  \tName of file to log results to. "
                                                               "If not specified, results printed at end of simulation." },
 {PROGRESS, 0, "",  "progress", option::Arg::None, "  --progress, \tSupply to show progress bar." },
 {UNKNOWN,  0, "" , ""   ,      option::Arg::None, "\nExamples:\n"
                                                   "  nengo_cpp --net basal_ganglia.net -t 1.0 --progress\n"
                                                   "  nengo_cpp --net spaun.net -t 7.5 --log spaun.h5\n" },
 {0,0,0,0,0,0}
};

void main(int argc, char **argv){

    argc -= (argc > 0); argv += (argc > 0); // skip program name argv[0] if present
    option::Stats  stats(serial_usage, argc, argv);
    option::Option options[stats.options_max], buffer[stats.buffer_max];
    option::Parser parse(serial_usage, argc, argv, options, buffer);

    if (parse.error()){
        return;
    }

    if (options[HELP] || argc == 0) {
        option::printUsage(std::cout, serial_usage);
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

    cout << "Building network..." << endl;
    bool use_mpi = false;
    auto sim = unique_ptr<Simulator>(new Simulator());
    sim->from_file(net_filename);

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
