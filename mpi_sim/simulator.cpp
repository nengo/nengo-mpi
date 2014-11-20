#include "simulator.hpp"

// The first chunk is kept on the master process,
// subsequent chunks will be sent out to other processes.
MpiSimulatorChunk* MpiSimulator::add_chunk(){
    MpiSimulatorChunk* chunk = new MpiSimulatorChunk();

    if(master_chunk == NULL){
        master_chunk = chunk;
    }else{
        remote_chunks.push_back(chunk);
    }

    return chunk;
}

MpiSimulator::MpiSimulator():
    master_chunk(NULL){
}

void MpiSimulator::finalize(){
    probe_counts[0] = master_chunk->get_num_probes();

    key_type probe_key;
    map<key_type, Probe<Matrix>*>::const_iterator probe_it = master_chunk->probe_map.begin();

    // Hook the probes in the master chunk into the probe_map
    for(; probe_it != master_chunk->probe_map.end(); probe_it++){
        probe_key = probe_it->first;
        probe_data[probe_key] = probe_it->second->get_data();
    }

    int chunk_index = 1;
    list<MpiSimulatorChunk*>::const_iterator it;

    for(it = remote_chunks.begin(); it != remote_chunks.end(); ++it){
        probe_counts[chunk_index] = (*it)->get_num_probes();
        chunk_index++;
    }

    if(!remote_chunks.empty()){
        mpi_interface.initialize_chunks(master_chunk, remote_chunks);
    }else{
        cout << "C++: Only one chunk supplied. Simulations will not use MPI." << endl;
    }
}

void MpiSimulator::run_n_steps(int steps){

    if(remote_chunks.empty()){
        master_chunk->run_n_steps(steps);
    }else{
        mpi_interface.run_n_steps(steps);
        mpi_interface.gather_probe_data(probe_data, probe_counts);
        mpi_interface.finish_simulation();
    }
}

vector<Matrix*>* MpiSimulator::get_probe_data(key_type probe_key){
    return probe_data.at(probe_key);
}

void MpiSimulator::write_to_file(string filename){
    ofstream ofs(filename);

    boost::archive::text_oarchive oa(ofs);
    oa << *this;
}

void MpiSimulator::read_from_file(string filename){
    ifstream ifs(filename);

    boost::archive::text_iarchive ia(ifs);
    ia >> *this;
}

string MpiSimulator::to_string() const{
    stringstream out;

    out << "<MpiSimulator" << endl;

    out << "**master chunk**" << endl;
    out << master_chunk << endl;

    out << "**remote chunks**" << endl;
    list<MpiSimulatorChunk*>::const_iterator it;
    for(it = remote_chunks.begin(); it != remote_chunks.end(); ++it){
        out << (**it) << endl;
    }

    return out.str();
}
