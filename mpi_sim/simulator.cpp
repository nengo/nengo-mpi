#include "simulator.hpp"

MpiSimulator::MpiSimulator():
    num_components(0), dt(0.001), master_chunk(NULL){
}

MpiSimulator::MpiSimulator(int num_components, float dt):
    num_components(num_components), dt(dt), master_chunk(NULL){

    master_chunk = new MpiSimulatorChunk("Chunk 0", dt);

    if(num_components == 1){
        cout << "Master: Only one chunk supplied. Simulations will not use MPI." << endl;
    }else{
        mpi_interface.initialize_chunks(master_chunk, num_components - 1);
    }

    for(int i = 0; i < num_components; i++){
        probe_counts[i] = 0;
    }
}

void MpiSimulator::finalize(){
    if(num_components > 1){
        mpi_interface.finalize();
    }
}

void MpiSimulator::run_n_steps(int steps){

    if(num_components == 1){
        master_chunk->run_n_steps(steps);
    }else{
        mpi_interface.run_n_steps(steps);
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
    if(component == 0){
        master_chunk->add_base_signal(key, label, *data);
    }else{
        mpi_interface.add_base_signal(component, key, label, data);
    }
}

void MpiSimulator::add_op(int component, string op_string){
    if(component == 0){
        master_chunk->add_op(op_string);
    }else{
        mpi_interface.add_op(component, op_string);
    }
}

void MpiSimulator::add_probe(int component, key_type probe_key, string signal_string, int period){
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
