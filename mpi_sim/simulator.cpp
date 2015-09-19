#include "simulator.hpp"

Simulator::Simulator(bool collect_timings)
:collect_timings(collect_timings){
    chunk = unique_ptr<MpiSimulatorChunk>(new MpiSimulatorChunk(collect_timings));
}

void Simulator::from_file(string filename){
    clock_t begin = clock();

    if(filename.length() == 0){
        stringstream s;
        s << "Got empty string for filename" << endl;
        throw runtime_error(s.str());
    }

    ifstream in_file(filename);

    if(!in_file.good()){
        stringstream s;
        s << "The network file " << filename << " does not exist." << endl;
        throw runtime_error(s.str());
    }

    in_file.close();

    // Use non-parallel property lists.
    hid_t file_plist = H5Pcreate(H5P_FILE_ACCESS);
    hid_t read_plist = H5Pcreate(H5P_DATASET_XFER);

    chunk->from_file(filename, file_plist, read_plist);

    for(const ProbeSpec& pi : chunk->probe_info){
        probe_data[pi.probe_key] = vector<unique_ptr<BaseSignal>>();
    }

    H5Pclose(file_plist);
    H5Pclose(read_plist);

    clock_t end = clock();
    cout << "Loading network from file took "
         << double(end - begin) / CLOCKS_PER_SEC << " seconds." << endl;
}

void Simulator::finalize_build(){
    chunk->finalize_build();
}

SignalView Simulator::get_signal(string signal_string){
    return chunk->get_signal_view(signal_string);
}

void Simulator::add_pyfunc(unique_ptr<Operator> pyfunc){
    chunk->add_op(move(pyfunc));
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
    cout << "Simulating " << steps << " steps took "
         << double(end - begin) / CLOCKS_PER_SEC << " seconds." << endl;
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
            "Calling get_probe_data, but probe data has been written to file.");
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

void Simulator::close(){
}

string Simulator::to_string() const{
    stringstream out;

    out << "<Simulator" << endl;

    out << "**chunk**" << endl;
    out << *chunk << endl;

    return out.str();
}
