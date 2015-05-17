#include <iostream>
#include <string>
#include <memory>
#include <vector>

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


enum serialOptionIndex {UNKNOWN, HELP, LOG, PROGRESS};

const option::Descriptor serial_usage[] =
{
 {UNKNOWN, 0, "" , "",          option::Arg::None, "USAGE: nengo_cpp [options] <network file> <simulation time> \n\n"
                                                   "where <network file> is a file specifying a network to simulate,\n"
                                                   "and <simulation time> is the amount of time to simulate the network\n"
                                                   "for, in seconds. Simulation results are logged to an HDF5 file.\n"
                                                   "Options:" },
 {HELP,     0, "" , "help",     option::Arg::None, "  --help  \tPrint usage and exit." },
 {PROGRESS, 0, "",  "progress", option::Arg::None, "  --progress  \tSupply to show progress bar." },
 {LOG,      0, "",  "log",      Arg::NonEmpty,     "  --log  \tName of file to log results to using HDF5. "
                                                               "If not specified, the log filename is the same as the "
                                                               "name of the network file, but with the .h5 extension."},
 {UNKNOWN,  0, "" , ""   ,      option::Arg::None, "\nExamples:\n"
                                                   "  nengo_cpp --progress basal_ganglia.net 1.0\n"
                                                   "  nengo_cpp --log ~/spaun_results.h5 spaun.net 7.5\n" },
 {0,0,0,0,0,0}
};

int main(int argc, char **argv){

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

    bool show_progress = bool(options[PROGRESS]);

    string log_filename;
    if(options[LOG]){
        log_filename = options[LOG].arg;
    }else{
        log_filename = net_filename.substr(0, net_filename.find_last_of(".")) + ".h5";
    }
    cout << "Will write simulation results to " << log_filename << endl;

    cout << "Building network..." << endl;
    auto sim = unique_ptr<Simulator>(new Simulator);
    sim->from_file(net_filename);

    cout << "Done building network..." << endl;

    cout << "dt: " << sim->dt << endl;
    int num_steps = int(sim_length / sim->dt);

    cout << "Num steps: " << num_steps << endl;
    cout << "Running simulation..." << endl;

    sim->run_n_steps(num_steps, show_progress, log_filename);

    return 0;
}
