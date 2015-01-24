#include "simulator.hpp"

char MpiSimulator::delim = '|';

MpiSimulator::MpiSimulator():
    num_components(0), dt(0.001), master_chunk(NULL){
}

MpiSimulator::MpiSimulator(int num_components, float dt, string out_filename):
    num_components(num_components), dt(dt), master_chunk(NULL), spawn(true){

    write_to_file = !(out_filename.empty());

    if (write_to_file){
        out_file = new ofstream(out_filename);
        (*out_file) << num_components << delim << dt << endl;
    }

    master_chunk = new MpiSimulatorChunk("Chunk 0", dt);

    if(num_components == 1){
        cout << "Master: Only one chunk supplied. Simulations will not use MPI." << endl;
    }else{
        mpi_interface.initialize_chunks(spawn, master_chunk, num_components - 1);
    }

    for(int i = 0; i < num_components; i++){
        probe_counts[i] = 0;
    }
}

MpiSimulator::MpiSimulator(string in_filename, bool spawn):
    num_components(0), dt(0.0), master_chunk(NULL), spawn(spawn){

    write_to_file = false;

    ifstream in_file(in_filename);

    string line;
    getline(in_file, line, delim);
    num_components = boost::lexical_cast<int>(line);

    getline(in_file, line, '\n');
    dt = boost::lexical_cast<float>(line);

    master_chunk = new MpiSimulatorChunk("Chunk 0", dt);

    if(num_components == 1){
        cout << "Master: Only one chunk supplied. Simulations will not use MPI." << endl;
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

        dbg(line << endl);
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

            BaseMatrix* data = new BaseMatrix(size1, size2);
            for(int i = 0; it < tokens.end(); it++, i++){
                (*data)(i / size2, i % size2) = boost::lexical_cast<floattype>(*it);
            }

            add_base_signal(component, key, label, data);
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

    if (write_to_file){
        out_file->close();
        delete out_file;
    }

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

    vector<BaseMatrix*> new_data;
    vector<BaseMatrix*> data;

    map<key_type, Probe*>::const_iterator probe_it = master_chunk->probe_map.begin();

    // Gather probe data from the master chunk
    for(; probe_it != master_chunk->probe_map.end(); probe_it++){

        data = probe_data.at(probe_it->first);
        new_data = probe_it->second->get_data();

        data.reserve(data.size() + new_data.size());
        data.insert(data.end(), new_data.begin(), new_data.end());
        probe_data[probe_it->first] = data;
    }
}

vector<key_type> MpiSimulator::get_probe_keys(){
    vector<key_type> keys;
    map<key_type, vector<BaseMatrix*>>::iterator it;
    for(it = probe_data.begin(); it != probe_data.end(); it++){
        keys.push_back(it->first);
    }

    return keys;
}

vector<BaseMatrix*> MpiSimulator::get_probe_data(key_type probe_key){
    return probe_data.at(probe_key);
}

void MpiSimulator::reset(){
    // TODO
    //Clear probe data
    //Tell master chunk to reset
    //Send a signal to remote chunks telling them to reset
}

void MpiSimulator::add_base_signal(int component, key_type key, string label, BaseMatrix* data){

    if(write_to_file){
        (*out_file) << "SIGNAL" << delim << component << delim << key << delim << label << delim << *data << endl;
    }

    if(component == 0){
        master_chunk->add_base_signal(key, label, *data);
    }else{
        mpi_interface.add_base_signal(component, key, label, data);
        delete data;
    }
}

void MpiSimulator::add_op(int component, string op_string){

    if(write_to_file){
        (*out_file) << "OP" << delim << component << delim << op_string << endl;
    }

    if(component == 0){
        master_chunk->add_op(op_string);
    }else{
        mpi_interface.add_op(component, op_string);
    }
}

void MpiSimulator::add_probe(int component, key_type probe_key, string signal_string, int period){

    if(write_to_file){
        (*out_file) << "PROBE" << delim << component << delim << probe_key << delim << signal_string << delim << period << endl;
    }

    if(component == 0){
        master_chunk->add_probe(probe_key, signal_string, period);
    }else{
        mpi_interface.add_probe(component, probe_key, signal_string, period);
    }

    probe_counts[component] += 1;
    probe_data[probe_key] = vector<BaseMatrix*>();
}

string MpiSimulator::to_string() const{
    stringstream out;

    out << "<MpiSimulator" << endl;

    out << "num_components: " << num_components << endl;
    out << "**master chunk**" << endl;
    out << *master_chunk << endl;

    return out.str();
}
