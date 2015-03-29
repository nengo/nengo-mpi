#include "simulator.hpp"

char MpiSimulator::delim = '|';

MpiSimulator::MpiSimulator()
:num_components(0), dt(0.001), master_chunk(NULL){}

MpiSimulator::MpiSimulator(int num_components, dtype dt)
:num_components(num_components), dt(dt), master_chunk(NULL), spawn(true){

    master_chunk = shared_ptr<MpiSimulatorChunk>(new MpiSimulatorChunk("Chunk 0", dt));

    if(num_components == 1){
        cout << "Master: Only one chunk supplied. "
             << "Simulations will not use MPI." << endl;
    }else{
        mpi_interface.initialize_chunks(spawn, master_chunk, num_components - 1);
    }

    for(int i = 0; i < num_components; i++){
        probe_counts[i] = 0;
    }
}

MpiSimulator::MpiSimulator(string in_filename, bool spawn)
:num_components(0), dt(0.001), master_chunk(NULL), spawn(spawn){
    ifstream in_file(in_filename);

    string line;
    getline(in_file, line, delim);
    num_components = boost::lexical_cast<int>(line);

    getline(in_file, line, '\n');
    dt = boost::lexical_cast<dtype>(line);

    master_chunk = shared_ptr<MpiSimulatorChunk>(new MpiSimulatorChunk("Chunk 0", dt));

    if(num_components == 1){
        cout << "Master: Only one chunk supplied. "
             << "Simulations will not use MPI." << endl;
    }else{
        mpi_interface.initialize_chunks(spawn, master_chunk, num_components - 1);
    }

    for(int i = 0; i < num_components; i++){
        probe_counts[i] = 0;
    }

    string cell;

    dbg("Loading nengo network from file.");

    while(getline(in_file, line, '\n')){
        stringstream line_stream;
        line_stream << line;
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
            boost::trim_if(cell, boost::is_any_of(",()[]"));
            boost::split(tokens, cell, boost::is_any_of(",()[]"), boost::token_compress_on);

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

            add_probe(component, probe_key, signal_string, period);
        }
    }

    in_file.close();

    finalize();
}

void MpiSimulator::finalize(){
    if(num_components > 1){
        mpi_interface.finalize();
    }
}

void MpiSimulator::run_n_steps(int steps, bool progress){
    if(num_components == 1){
        master_chunk->run_n_steps(steps, progress);
    }else{
        mpi_interface.run_n_steps(steps, progress);
        mpi_interface.gather_probe_data(probe_data, probe_counts);
        mpi_interface.finish_simulation();
    }

    // Gather probe data from the master chunk
    for(auto& kv: master_chunk->probe_map){
        auto& data = probe_data.at(kv.first);
        auto new_data = (kv.second)->harvest_data();

        data.reserve(data.size() + new_data.size());

        for(auto& nd : new_data){
            data.push_back(move(nd));
        }
    }
}

vector<key_type> MpiSimulator::get_probe_keys(){
    vector<key_type> keys;
    for(auto const& kv: probe_data){
        keys.push_back(kv.first);
    }

    return keys;
}

vector<unique_ptr<BaseSignal>> MpiSimulator::get_probe_data(key_type probe_key){
    return move(probe_data.at(probe_key));
}

void MpiSimulator::reset(){
    // TODO
    //Clear probe data
    //Tell master chunk to reset
    //Send a signal to remote chunks telling them to reset
}

void MpiSimulator::add_base_signal(
        int component, key_type key, string label, unique_ptr<BaseSignal> data){

    dbg("SIGNAL" << delim << component << delim
                 << key << delim << label << delim << *data);

    if(component == 0){
        master_chunk->add_base_signal(key, label, move(data));
    }else{
        mpi_interface.add_base_signal(component, key, label, move(data));
    }
}

SignalView MpiSimulator::get_signal_from_master(string signal_string){
    return master_chunk->get_signal_view(signal_string);
}

void MpiSimulator::add_op(int component, string op_string){

    dbg("OP" << delim << component << delim << op_string);

    if(component == 0){
        master_chunk->add_op(op_string);
    }else{
        mpi_interface.add_op(component, op_string);
    }
}

void MpiSimulator::add_op_to_master(unique_ptr<Operator> op){
    master_chunk->add_op(move(op));
}

void MpiSimulator::add_probe(
        int component, key_type probe_key, string signal_string, int period){

    dbg("PROBE" << delim << component << delim << probe_key << delim
                << signal_string << delim << period);

    if(component == 0){
        master_chunk->add_probe(probe_key, signal_string, period);
    }else{
        mpi_interface.add_probe(component, probe_key, signal_string, period);
    }

    probe_counts[component] += 1;
    probe_data[probe_key] = vector<unique_ptr<BaseSignal>>();
}

string MpiSimulator::to_string() const{
    stringstream out;

    out << "<MpiSimulator" << endl;

    out << "num_components: " << num_components << endl;
    out << "**master chunk**" << endl;
    out << *master_chunk << endl;

    return out.str();
}
