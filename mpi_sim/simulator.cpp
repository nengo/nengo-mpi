#include "simulator.hpp"

char Simulator::delim = '|';

Simulator::Simulator()
:Simulator(0.001){}

Simulator::Simulator(dtype dt)
:dt(dt), chunk(NULL){
    chunk = shared_ptr<MpiSimulatorChunk>(new MpiSimulatorChunk(0, "Chunk 0", dt));
}

void Simulator::add_base_signal(
        key_type key, string label, unique_ptr<BaseSignal> data){

    dbg("SIGNAL" << delim << key << delim << label << delim << *data);

    chunk->add_base_signal(key, label, move(data));
}

void Simulator::add_base_signal(
        int component, key_type key, string label, unique_ptr<BaseSignal> data){

    if(component != 0){
        throw logic_error(
            "Adding signal with non-zero component to non-mpi simulator.");
    }

    add_base_signal(key, label, move(data));
}

void Simulator::add_op(string op_string){

    dbg("OP" << delim << delim << op_string);

    chunk->add_op(op_string);
}

void Simulator::add_op(int component, string op_string){
    if(component != 0){
        throw logic_error(
            "Adding operator with non-zero component to non-mpi simulator.");
    }

    add_op(op_string);
}

void Simulator::add_probe(
        key_type probe_key, string signal_string, dtype period){

    stringstream ss;

    ss << "PROBE" << delim << 0 << delim << probe_key << delim
                  << signal_string << delim << period;

    probe_info.push_back(ss.str());

    dbg(ss.str());

    chunk->add_probe(probe_key, signal_string, period);
    probe_data[probe_key] = vector<unique_ptr<BaseSignal>>();
}

void Simulator::add_probe(
        int component, key_type probe_key, string signal_string, dtype period){

    if(component != 0){
        throw logic_error(
            "Adding probe with non-zero component to non-mpi simulator.");
    }

    add_probe(probe_key, signal_string, period);
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

    chunk->set_log_filename(log_filename);
    chunk->run_n_steps(steps, progress);

    gather_probe_data();
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
