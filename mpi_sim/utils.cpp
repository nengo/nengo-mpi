#include "utils.hpp"

unique_ptr<Simulator> create_simulator_from_file(string filename){
    char delim = Simulator::delim;

    ifstream in_file(filename);

    if(!in_file.good()){
        stringstream s;
        s << "The network file " << filename << " does not exist." << endl;
        throw runtime_error(s.str());
    }

    string line;
    getline(in_file, line, delim);

    int num_components = boost::lexical_cast<int>(line);

    getline(in_file, line, '\n');
    dtype dt = boost::lexical_cast<dtype>(line);

    unique_ptr<Simulator> simulator;

    if(num_components == 1){
        simulator = unique_ptr<Simulator>(new Simulator(dt));
    }else{
        bool spawn = false;
        simulator = unique_ptr<Simulator>(new MpiSimulator(num_components, dt, spawn));
    }

    string cell;

    dbg("Loading nengo network from file.");

    while(getline(in_file, line, '\n')){
        stringstream line_stream;
        line_stream << line;
        dbg("Reading: " << line_stream.str() << endl);

        getline(line_stream, cell, delim);

        if(cell.compare("SIGNAL") == 0){
            getline(line_stream, cell, delim);
            int component = boost::lexical_cast<int>(cell);

            getline(line_stream, cell, delim);
            key_type key = boost::lexical_cast<key_type>(cell);

            getline(line_stream, cell, delim);
            string label = cell;

            getline(line_stream, cell, delim);

            vector<string> tokens;
            boost::trim_if(cell, boost::is_any_of(",()[] "));
            boost::split(tokens, cell, boost::is_any_of(",()[] "), boost::token_compress_on);

            vector<string>::iterator it = tokens.begin();
            int size1 = boost::lexical_cast<int>(*(it++));
            int size2 = boost::lexical_cast<int>(*(it++));

            auto data = unique_ptr<BaseSignal>(new BaseSignal(size1, size2));

            for(int i = 0; it < tokens.end(); it++, i++){
                (*data)(i / size2, i % size2) = boost::lexical_cast<dtype>(*it);
            }

            simulator->add_base_signal(component, key, label, move(data));

        }else if(cell.compare("OP") == 0){
            getline(line_stream, cell, delim);
            int component = boost::lexical_cast<int>(cell);

            getline(line_stream, cell, delim);
            string op_string = cell;

            simulator->add_op(component, op_string);

        }else if(cell.compare("PROBE") == 0){

            getline(line_stream, cell, delim);
            int component = boost::lexical_cast<int>(cell);

            getline(line_stream, cell, delim);
            key_type probe_key = boost::lexical_cast<key_type>(cell);

            getline(line_stream, cell, delim);
            string signal_string = cell;

            getline(line_stream, cell, delim);
            int period = boost::lexical_cast<int>(cell);

            getline(line_stream, cell, delim);
            string name = cell;

            simulator->add_probe(component, probe_key, signal_string, period, name);
        }
    }

    in_file.close();

    simulator->finalize_build();

    return simulator;
}
