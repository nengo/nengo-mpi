#include "simulator.hpp"

char Simulator::delim = '|';

Simulator::Simulator()
:dt(0.001){}

Simulator::Simulator(dtype dt)
:dt(dt), chunk(NULL){
    init(dt);
}

void Simulator::init(dtype dt){
    cout << "In simulator.init" << endl;
    chunk = shared_ptr<MpiSimulatorChunk>(new MpiSimulatorChunk(0, "Chunk 0", dt, 1));
}

void Simulator::add_base_signal(
        key_type key, string label, unique_ptr<BaseSignal> data){

    dbg("SIGNAL" << delim << 0 << delim << key << delim << label << delim << *data);

    chunk->add_base_signal(key, label, move(data));
}

void Simulator::add_base_signal(
        int component, key_type key, string label, unique_ptr<BaseSignal> data){

    add_base_signal(key, label, move(data));
}

void Simulator::add_op(string op_string){

    dbg("OP" << delim << 0 << delim << op_string);

    chunk->add_op(op_string);
}

void Simulator::add_op(int component, string op_string){

    add_op(op_string);
}

void Simulator::add_probe(
        key_type probe_key, string signal_string, dtype period, string name){

    stringstream ss;

    ss << "PROBE" << delim << 0 << delim << probe_key << delim
                  << signal_string << delim << period << delim << name;

    probe_info.push_back(ss.str());

    dbg(ss.str());

    chunk->add_probe(probe_key, signal_string, period);
    probe_data[probe_key] = vector<unique_ptr<BaseSignal>>();
}

void Simulator::add_probe(
        int component, key_type probe_key, string signal_string, dtype period, string name){

    add_probe(probe_key, signal_string, period, name);
}

SignalView Simulator::get_signal(string signal_string){
    return chunk->get_signal_view(signal_string);
}

void Simulator::add_op(unique_ptr<Operator> op){
    chunk->add_op(move(op));
}

void Simulator::finalize_build(){

    chunk->set_simulation_log(
        unique_ptr<SimulationLog>(new SimulationLog(probe_info, dt)));
}

void Simulator::run_n_steps(int steps, bool progress, string log_filename){

    clock_t begin = clock();

    chunk->set_log_filename(log_filename);
    chunk->run_n_steps(steps, progress);

    if(!chunk->is_logging()){
        gather_probe_data();
    }

    chunk->close_simulation_log();

    clock_t end = clock();
    cout << "Simulating " << steps << " steps took " << double(end - begin) / CLOCKS_PER_SEC << " seconds." << endl;
}

void Simulator::gather_probe_data(){
    // Gather probe data from the chunk
    for(auto& kv: chunk->probe_map){
        auto& data = probe_data.at(kv.first);
        auto new_data = (kv.second)->harvest_data();

        data.reserve(data.size() + new_data.size());

        for(auto& nd : new_data){
            data.push_back(move(nd));
        }
    }
}

vector<unique_ptr<BaseSignal>> Simulator::get_probe_data(key_type probe_key){
    if(chunk->is_logging()){
        throw logic_error(
            "Calling get_probe_data, but probe data has all been written to file");
    }

    return move(probe_data.at(probe_key));
}

vector<key_type> Simulator::get_probe_keys(){
    vector<key_type> keys;
    for(auto const& kv: probe_data){
        keys.push_back(kv.first);
    }

    return keys;
}

void Simulator::reset(){
    // TODO
    //Clear probe data
    //Send a signal to remote chunks telling them to reset
}

string Simulator::to_string() const{
    stringstream out;

    out << "<Simulator" << endl;

    out << "**chunk**" << endl;
    out << *chunk << endl;

    return out.str();
}

void Simulator::from_file(string filename){
    char delim = Simulator::delim;

    ifstream in_file(filename);

    if(!in_file.good()){
        stringstream s;
        s << "The network file " << filename << " does not exist." << endl;
        throw runtime_error(s.str());
    }

    cout << "Loading nengo network from file." << endl;

    string line;
    getline(in_file, line, delim);

    int n_components = boost::lexical_cast<int>(line);

    cout << "Network has " << n_components << " components." << endl;

    getline(in_file, line, '\n');
    dt = boost::lexical_cast<dtype>(line);
    init(dt);

    string cell;

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

            add_base_signal(component, key, label, move(data));

        }else if(cell.compare("OP") == 0){
            getline(line_stream, cell, delim);
            int component = boost::lexical_cast<int>(cell);

            getline(line_stream, cell, delim);
            string op_string = cell;

            add_op(component, op_string);

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

            add_probe(component, probe_key, signal_string, period, name);
        }
    }

    in_file.close();

    finalize_build();
}
